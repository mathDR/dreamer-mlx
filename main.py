"""Dreamer-style training from pixels on dm_control's acrobot (swingup), in Apple MLX.

Loop (as in Dreamer):
    1. prefill the replay buffer with bang-bang random episodes
    2. pretrain the world model on that data
    3. repeat: collect one episode with the current actor, then do a batch of
       gradient steps, each consisting of
         a. a world-model update on replayed (image, action, reward, cont) sequences
         b. actor / critic updates on imagined rollouts started from the posterior
            states of that same batch. The actor backpropagates the lambda-return
            through the imagined dynamics (Dreamer V1/V2 style); the critic regresses
            onto lambda-return targets with a two-hot cross-entropy.

After pretraining, ``imagination_check`` compares open-loop dreams against real frames.
The logged ``kl`` is the raw (unclamped) estimate; it should stay positive. ``pstd`` is the
mean posterior std; if it sits at the floor (--post-min-std) the posterior has gone
near-deterministic.

Posterior: a Gaussian encoder head q(z | h, image) = N(mean, std); prior: a conditional
normalizing flow p(z | h). KL = -H[q] - log p_flow(z | h), with z a reparameterized sample.

Rendering: macOS works out of the box. On headless Linux set MUJOCO_GL=egl or osmesa
before running.

    python main.py                      # defaults
    python main.py --episodes 100 --train-steps 100
"""

import argparse
import os
import time

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn

from dreamer_mlx.policies import ActorCriticUpdater, ContinuousActor, ContinuousCritic
from dreamer_mlx.simulator.generate_data import (
    ACTION_REPEAT,
    IMAGE_SIZE,
    BangBangPolicy,
    collect_episode,
    make_buffer,
    make_env,
    prefill,
    render,
)
from dreamer_mlx.utils import TwoHotDist, symexp, symlog
from dreamer_mlx.world_model import FlowRSSM
from dreamer_mlx.world_model.codecs import DreamerV3Encoder, ImageDecoder

# --------------------------------------------------------------------------- #
# World model over pixels
# --------------------------------------------------------------------------- #
class ImageWorldModel(nn.Module):
    """FlowRSSM + conv encoder/decoder + reward and continue heads.

    Same structure and loss as ``ContinuousDreamerWorldModel`` but with image
    observations. Exposes ``rssm``, ``reward_predictor`` and ``continue_predictor`` so
    it can be dropped into ``ActorCriticUpdater``.
    """

    def __init__(
        self,
        action_dim: int,
        deter_dim: int = 256,
        latent_dim: int = 32,
        cnn_depth: int = 32,
        num_bins: int = 255,
        image_size: int = IMAGE_SIZE,
        recon_weight: float = 100.0,
        kl_weight: float = 0.1,
        free_nats: float = 0.5,
        min_std: float = 0.1,
        max_std: float = 1.0,
        std_activation: str = "softplus",
    ):
        super().__init__()
        feature_dim = deter_dim + latent_dim
        self.recon_weight, self.kl_weight, self.free_nats = recon_weight, kl_weight, free_nats

        # Prior: conditional flow p(z | h). Posterior: Gaussian q(z | h, image) from the encoder.
        self.rssm = FlowRSSM(action_dim, deter_dim, latent_dim, flow_posterior=False)
        self.encoder = DreamerV3Encoder(
            image_shape=(image_size, image_size, 3),
            cnn_depth=cnn_depth,
            latent_dim=latent_dim,
            context_dim=deter_dim,
            min_std=min_std,
            max_std=max_std,
            std_activation=std_activation,
        )
        self.decoder = ImageDecoder(latent_dim=feature_dim, out_channels=3, depth=cnn_depth)
        self.reward_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, num_bins)
        )
        self.continue_predictor = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, 1)
        )

    def embed(self, image: mx.array) -> mx.array:
        """(..., H, W, 3) uint8 -> (..., encoder.embed_dim) deterministic backbone features."""
        return self.encoder.features(image=image)

    def loss(self, image, actions, rewards, continues):
        """Returns (total, recon, reward, continue, kl, features, post_std).

        Only ``total`` is differentiated by ``nn.value_and_grad``; the rest ride along
        as auxiliary outputs. ``features`` are the posterior states [deter, latent],
        reused as imagination start points.
        """
        B, T = image.shape[:2]
        rewards = mx.squeeze(rewards, axis=-1)
        continues = mx.squeeze(continues, axis=-1)
        embeds = self.embed(image)

        deter = mx.zeros((B, self.rssm.deter_dim))
        latent = mx.zeros((B, self.rssm.latent_dim))
        features, kl, post_std = [], 0.0, 0.0
        for t in range(T):
            prev_act = actions[:, t - 1] if t > 0 else mx.zeros_like(actions[:, 0])
            deter, latent, mean, std = self.rssm.observe_step_gaussian(
                latent, prev_act, deter, embeds[:, t], self.encoder
            )
            # KL(N(mean, std) || flow prior): analytic entropy + flow log-prob at the sample.
            kl = kl + mx.mean(self.rssm.gaussian_kl(mean, std, latent, deter))
            post_std = post_std + mx.mean(std)
            features.append(mx.concatenate([deter, latent], axis=-1))
        features = mx.stack(features, axis=1)  # (B, T, F)
        kl_raw = kl / T  # logged; can dip below the free-nats floor
        post_std = post_std / T
        kl = mx.maximum(kl_raw, self.free_nats)

        flat = features.reshape(B * T, -1)
        pred = self.decoder(flat)
        target = image.reshape(B * T, *image.shape[2:]).astype(mx.float32) / 255.0
        recon_loss = mx.mean((pred - target) ** 2)

        reward_dist = TwoHotDist(self.reward_predictor(features))
        reward_loss = mx.mean((reward_dist.mean() - symlog(rewards)) ** 2)

        cont_logits = mx.squeeze(self.continue_predictor(features), axis=-1)
        cont_loss = mx.mean(nn.losses.binary_cross_entropy(cont_logits, continues))

        total = self.recon_weight * recon_loss + reward_loss + cont_loss + self.kl_weight * kl
        return total, recon_loss, reward_loss, cont_loss, kl_raw, features, post_std


# --------------------------------------------------------------------------- #
# Acting
# --------------------------------------------------------------------------- #
class LearnedPolicy:
    """Actor wrapped in the ``policy() / policy.reset()`` interface ``collect_episode`` expects.

    ``collect_episode`` calls ``policy()`` with no arguments, so the policy renders the
    current frame from the env itself, filters it into the RSSM state, and samples an
    action from the actor.
    """

    def __init__(self, env, wm, actor, low, high):
        self.env, self.wm, self.actor = env, wm, actor
        self.low, self.high = low, high
        self.reset()

    def reset(self):
        self.deter = mx.zeros((1, self.wm.rssm.deter_dim))
        self.latent = mx.zeros((1, self.wm.rssm.latent_dim))
        self.prev_action = mx.zeros((1, self.low.shape[0]))

    def __call__(self) -> np.ndarray:
        frame = mx.array(np.ascontiguousarray(render(self.env)))[None]  # (1, H, W, 3) uint8
        features = self.wm.embed(frame)
        self.deter, self.latent, _, _ = self.wm.rssm.observe_step_gaussian(
            self.latent, self.prev_action, self.deter, features, self.wm.encoder
        )
        feat = mx.concatenate([self.deter, self.latent], axis=-1)
        action, _, _ = self.actor.sample_and_log_prob(feat)  # in [-1, 1]
        mx.eval(action, self.deter, self.latent)
        self.prev_action = action
        a = np.asarray(action)[0]
        return (self.low + (a + 1.0) * 0.5 * (self.high - self.low)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Actor / critic objectives (replace ActorCriticUpdater.calculate_losses)
# --------------------------------------------------------------------------- #
CRITIC_BINS = mx.linspace(-20.0, 20.0, 255)  # symlog space, ~0.16 per bin


def two_hot(x, bins):
    """x: (...,) -> (..., n_bins) two-hot probabilities over uniformly spaced bins."""
    n = bins.shape[0]
    step = (bins[-1] - bins[0]) / (n - 1)
    x = mx.clip(x, bins[0], bins[-1])
    pos = (x - bins[0]) / step
    lo = mx.clip(mx.floor(pos), 0, n - 2)
    w_hi = pos - lo
    idx = mx.arange(n)
    lo_i = lo.astype(mx.int32)[..., None]
    return (1 - w_hi)[..., None] * (idx == lo_i) + w_hi[..., None] * (idx == lo_i + 1)


def imagined_returns(wm, critic, feats, gamma, lam):
    """Lambda-returns over imagined features (N, H, F) -> (N, H-1)."""
    rewards = symexp(TwoHotDist(wm.reward_predictor(feats)).mean())          # (N, H)
    conts = mx.sigmoid(wm.continue_predictor(feats))[..., 0]                 # (N, H)
    values = symexp(critic(feats)[0])                                        # (N, H)
    H = feats.shape[1]
    rets, nxt = [None] * (H - 1), values[:, -1]
    for t in reversed(range(H - 1)):
        nxt = rewards[:, t] + gamma * conts[:, t] * ((1 - lam) * values[:, t + 1] + lam * nxt)
        rets[t] = nxt
    return mx.stack(rets, axis=1)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def imagination_check(wm, buffer, norm_action, ctx=10, horizon=15):
    """Filter `ctx` real frames, then dream `horizon` steps open-loop and compare to reality.

    Prints the raw (unclamped) KL, per-step dream MSE, and a copy-last-frame baseline.
    If dream MSE is no better than the baseline, the prior hasn't learned dynamics.
    """
    batch = buffer.sample_sequences(1, ctx + horizon)
    image = batch["image"]  # (1, L, H, W, 3) uint8
    actions = norm_action(batch["action"])  # (1, L, A)
    embeds = wm.embed(image)

    deter = mx.zeros((1, wm.rssm.deter_dim))
    latent = mx.zeros((1, wm.rssm.latent_dim))
    post_feats, kls = [], []
    for t in range(ctx):  # posterior filtering on real frames
        prev_a = actions[:, t - 1] if t > 0 else mx.zeros_like(actions[:, 0])
        deter, latent, mean, std = wm.rssm.observe_step_gaussian(
            latent, prev_a, deter, embeds[:, t], wm.encoder
        )
        kls.append(mx.mean(wm.rssm.gaussian_kl(mean, std, latent, deter)))
        post_feats.append(mx.concatenate([deter, latent], axis=-1))

    dream_feats = []
    for t in range(ctx, ctx + horizon):  # open-loop: prior only, real actions
        deter, latent = wm.rssm.imagine_step(latent, actions[:, t - 1], deter)
        dream_feats.append(mx.concatenate([deter, latent], axis=-1))

    def decode(feats):
        return wm.decoder(mx.concatenate(feats, axis=0))  # (N, H, W, 3) in [0, 1]

    real = image[0].astype(mx.float32) / 255.0
    post_mse = mx.mean((decode(post_feats) - real[:ctx]) ** 2)
    dream_err = mx.mean((decode(dream_feats) - real[ctx:]) ** 2, axis=(1, 2, 3))
    base_err = mx.mean((real[ctx - 1] - real[ctx:]) ** 2, axis=(1, 2, 3))
    mx.eval(post_mse, dream_err, base_err, kls)

    print(f"  raw KL (unclamped): {np.mean([float(k.item()) for k in kls]):.3f}")
    print(f"  posterior recon MSE (context): {post_mse.item():.5f}")
    print("  step | dream MSE | copy-last-frame baseline")
    for i in (0, 4, 9, horizon - 1):
        if i < horizon:
            print(f"  {i + 1:4d} | {dream_err[i].item():.5f}   | {base_err[i].item():.5f}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200, help="on-policy episodes after prefill")
    p.add_argument("--prefill-episodes", type=int, default=40)
    p.add_argument("--pretrain-steps", type=int, default=200, help="world-model-only steps")
    p.add_argument("--train-steps", type=int, default=100, help="gradient steps per episode")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--imag-starts", type=int, default=512, help="imagination start states")
    p.add_argument("--capacity", type=int, default=50_000)
    p.add_argument("--deter-dim", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--cnn-depth", type=int, default=32)
    p.add_argument("--recon-weight", type=float, default=1000.0)
    p.add_argument("--kl-weight", type=float, default=0.1)
    p.add_argument("--free-nats", type=float, default=0.5)
    p.add_argument("--max-std", type=float, default=0.5, help="actor std cap (pre-tanh)")
    p.add_argument("--post-min-std", type=float, default=0.1, help="posterior std floor")
    p.add_argument("--post-max-std", type=float, default=1.0, help="posterior std cap (sigmoid only)")
    p.add_argument("--post-std-act", choices=["softplus", "sigmoid"], default="softplus")
    p.add_argument("--ent-coef", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    # ---- environment, buffer, random policy ----
    env = make_env("acrobot", "swingup", seed=args.seed)
    spec = env.action_spec()
    low = np.asarray(spec.minimum, np.float32).ravel()
    high = np.asarray(spec.maximum, np.float32).ravel()
    action_dim = low.shape[0]
    steps_per_episode = int(env._step_limit) // ACTION_REPEAT

    buffer = make_buffer(env, capacity=args.capacity, seed=args.seed)
    random_policy = BangBangPolicy(spec, rng=np.random.default_rng(args.seed))

    def norm_action(a):  # env action range -> [-1, 1] (what the actor emits)
        return 2.0 * (a - mx.array(low)) / mx.array(high - low) - 1.0

    # ---- models ----
    wm = ImageWorldModel(
        action_dim, args.deter_dim, args.latent_dim, cnn_depth=args.cnn_depth,
        recon_weight=args.recon_weight, kl_weight=args.kl_weight, free_nats=args.free_nats,
        min_std=args.post_min_std, max_std=args.post_max_std, std_activation=args.post_std_act,
    )
    feat_dim = args.deter_dim + args.latent_dim
    actor = ContinuousActor(feature_dim=feat_dim, action_dim=action_dim, max_std=args.max_std)
    critic = ContinuousCritic(feature_dim=feat_dim)
    critic.bins = CRITIC_BINS  # finer support than the default +/-100
    critic.freeze(keys=["bins"])  # fixed support, not a learnable parameter
    updater = ActorCriticUpdater(actor, critic, wm, gamma=0.99, lam=0.95, entropy_coef=1e-2)

    wm_opt = optim.Adam(learning_rate=3e-4)
    actor_opt = optim.Adam(learning_rate=3e-5)
    critic_opt = optim.Adam(learning_rate=1e-3)

    GAMMA, LAM, ENT_COEF, HORIZON = 0.99, 0.95, args.ent_coef, 12

    def actor_loss_fn(actor_, start):
        # Backprop the lambda-return through the imagined dynamics (Dreamer V1/V2 style).
        feats = updater.rollout_imagination(start, horizon=HORIZON)          # (N, H, F)
        rets = imagined_returns(wm, critic, feats, GAMMA, LAM)
        entropy = mx.mean(actor_.sample_and_log_prob(mx.stop_gradient(feats))[2])
        return -mx.mean(rets) - ENT_COEF * entropy, mx.stop_gradient(feats), entropy

    def critic_loss_fn(critic_, feats):
        # feats are detached imagined states; regress the critic onto lambda-return targets.
        rets = mx.stop_gradient(imagined_returns(wm, critic_, feats, GAMMA, LAM))
        logits = critic_(feats)[1][:, :-1]
        target = two_hot(symlog(rets), CRITIC_BINS)
        return mx.mean(nn.losses.cross_entropy(logits, target))

    # nn.value_and_grad differentiates only its model's parameters and only the first
    # element of a returned tuple; the extra outputs (feats, entropy) ride along.
    wm_grad_fn = nn.value_and_grad(wm, lambda m, *a: m.loss(*a))
    actor_grad_fn = nn.value_and_grad(actor, actor_loss_fn)
    critic_grad_fn = nn.value_and_grad(critic, critic_loss_fn)

    def train_step(train_ac: bool):
        batch = buffer.sample_sequences(args.batch_size, args.seq_len)
        image = batch["image"]  # (B, L, H, W, 3) uint8
        actions = norm_action(batch["action"])

        (total, recon, rew, cont, kl, feats, pstd), grads = wm_grad_fn(
            wm, image, actions, batch["reward"], batch["cont"]
        )
        grads, _ = optim.clip_grad_norm(grads, 100.0)
        wm_opt.update(wm, grads)

        metrics = {"wm": total, "recon": recon, "rew": rew, "kl": kl, "pstd": pstd}
        if train_ac:
            start = mx.stop_gradient(feats.reshape(-1, feat_dim))
            n = min(args.imag_starts, start.shape[0])
            start = start[mx.random.permutation(start.shape[0])[:n]]

            (a_loss, imag_feats, ent), a_grads = actor_grad_fn(actor, start)
            c_loss, c_grads = critic_grad_fn(critic, imag_feats)
            a_grads, _ = optim.clip_grad_norm(a_grads, 10.0)
            c_grads, _ = optim.clip_grad_norm(c_grads, 10.0)
            actor_opt.update(actor, a_grads)
            critic_opt.update(critic, c_grads)
            metrics.update(actor=a_loss, critic=c_loss, ent=ent)

        mx.eval(wm.parameters(), actor.parameters(), critic.parameters(),
                wm_opt.state, actor_opt.state, critic_opt.state, metrics)
        return {k: float(v.item()) for k, v in metrics.items()}

    def save(tag=""):
        os.makedirs(args.out_dir, exist_ok=True)
        wm.save_weights(os.path.join(args.out_dir, f"world_model{tag}.safetensors"))
        actor.save_weights(os.path.join(args.out_dir, f"actor{tag}.safetensors"))
        critic.save_weights(os.path.join(args.out_dir, f"critic{tag}.safetensors"))

    # ---- 1. prefill ----
    print(f"Prefilling with {args.prefill_episodes} random episodes "
          f"({steps_per_episode} agent steps each)...")
    prefill(env, buffer, random_policy, num_episodes=args.prefill_episodes)

    # ---- 2. world-model pretraining ----
    print(f"Pretraining world model for {args.pretrain_steps} steps...")
    for i in range(args.pretrain_steps):
        m = train_step(train_ac=False)
        if (i + 1) % 25 == 0:
            print(f"  pretrain {i + 1:4d} | wm {m['wm']:.3f} recon {m['recon']:.4f} "
                  f"rew {m['rew']:.4f} kl {m['kl']:.3f} pstd {m['pstd']:.3f}")

    print("Imagination check (open-loop dreams vs. real frames):")
    imagination_check(wm, buffer, norm_action)

    # ---- 3. interleaved collect / train ----
    agent = LearnedPolicy(env, wm, actor, low, high)
    for ep in range(1, args.episodes + 1):
        t0 = time.time()
        ep_return, n_steps = collect_episode(env, buffer, agent)
        t_collect = time.time() - t0

        t0 = time.time()
        ms = [train_step(train_ac=True) for _ in range(args.train_steps)]
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        t_train = time.time() - t0

        print(f"Episode {ep:4d} | return {ep_return:7.2f} | buffer {len(buffer):6d} | "
              f"wm {avg['wm']:.3f} recon {avg['recon']:.4f} rew {avg['rew']:.4f} "
              f"kl {avg['kl']:.2f} pstd {avg['pstd']:.2f} | actor {avg['actor']:+.3f} critic {avg['critic']:.3f} ent {avg['ent']:+.2f} | "
              f"{t_collect:.0f}s collect, {t_train:.0f}s train")

        if ep % 25 == 0:
            save()

    save()
    print(f"Saved world_model / actor / critic .safetensors to '{args.out_dir}'")


if __name__ == "__main__":
    main()
