from typing import Protocol

import mlx.core as mx
from mlx import nn

from dreamer_mlx.utils import TwoHotDist, gaussian_entropy, gaussian_sample, symlog

# Recurrent state tuple for mLSTM: (C_t, n_t, m_t)
type MLSTMState = tuple[mx.array, mx.array, mx.array]

class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim

        # Equinox packs the 3 gate projections into a single weight matrix for performance.
        # We do the same here using MLX Linear layers.
        self.weight_ih = nn.Linear(input_dim, 3 * hidden_dim, bias=bias)
        self.weight_hh = nn.Linear(hidden_dim, 3 * hidden_dim, bias=bias)

    def __call__(self, x: mx.array, h: mx.array) -> mx.array:
        """
        Processes a single step.
        x: shape (input_dim,) or (batch_size, input_dim)
        h: shape (hidden_dim,) or (batch_size, hidden_dim)
        """
        # Project inputs and previous hidden state
        gates_x = self.weight_ih(x)
        gates_h = self.weight_hh(h)

        # Split into reset (r), update (z), and candidate (n) gates
        r_x, z_x, n_x = mx.split(gates_x, 3, axis=-1)
        r_h, z_h, n_h = mx.split(gates_h, 3, axis=-1)

        # Apply gate activations
        r = mx.sigmoid(r_x + r_h)
        z = mx.sigmoid(z_x + z_h)
        
        # Calculate new candidate hidden state
        n = mx.tanh(n_x + r * n_h)

        # Compute next hidden state
        next_h = (1 - z) * n + z * h
        return next_h

# =====================================================================
# RECURRENT STATE-SPACE MODEL WITH CONDITIONAL FLOW LATENTS
# =====================================================================

class ConditionalAffineCoupling(nn.Module):
    def __init__(self, latent_dim, context_dim, hidden_dim=128):
        super().__init__()
        self.half_dim = latent_dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.half_dim + context_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, z, context, reverse=False):
        z1, z2 = z[..., :self.half_dim], z[..., self.half_dim:]
        net_input = mx.concatenate([z1, context], axis=-1)
        out = self.net(net_input)
        s, t = out[..., :self.half_dim], out[..., self.half_dim:]
        s = mx.tanh(s)
        if not reverse:
            return mx.concatenate([z1, z2 * mx.exp(s) + t], axis=-1), mx.sum(s, axis=-1)
        else:
            return mx.concatenate([z1, (z2 - t) * mx.exp(-s)], axis=-1), mx.sum(-s, axis=-1)

class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, latent_dim, context_dim, num_layers=4):
        super().__init__()
        self.layers = [ConditionalAffineCoupling(latent_dim, context_dim) for _ in range(num_layers)]
        self.latent_dim = latent_dim

    def log_prob(self, z, context):
        log_det_total = 0.0
        current_z = z
        for i, layer in enumerate(self.layers):
            if i % 2 == 1:
                current_z = mx.concatenate([current_z[..., current_z.shape[-1]//2:], current_z[..., :current_z.shape[-1]//2]], axis=-1)
            current_z, log_det = layer.forward(current_z, context, reverse=False)
            log_det_total += log_det
        base_log_prob = -0.5 * mx.sum(current_z**2 + mx.log(2 * mx.pi), axis=-1)
        return base_log_prob + log_det_total

    def sample(self, context):
        current_z = mx.random.normal(shape=context.shape[:-1] + (self.latent_dim,))
        for i, layer in enumerate(reversed(self.layers)):
            current_z, _ = layer.forward(current_z, context, reverse=True)
            if (len(self.layers) - 1 - i) % 2 == 1:
                current_z = mx.concatenate([current_z[..., current_z.shape[-1]//2:], current_z[..., :current_z.shape[-1]//2]], axis=-1)
        return current_z


class PosteriorEncoder(Protocol):
    def posterior(self, features: mx.array, context: mx.array) -> tuple[mx.array, mx.array]:
        ...


class mLSTMCell(nn.Module):
    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        d_model = num_heads * head_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.i_gate = nn.Linear(d_model, num_heads)
        self.f_gate = nn.Linear(d_model, num_heads)
        self.o_gate = nn.Linear(d_model, d_model)

        self.out_norm = nn.LayerNorm(head_dim)

    def _norm_heads(self, h: mx.array) -> mx.array:
        return self.out_norm(h).reshape(*h.shape[:-2], self.num_heads * self.head_dim)

    def init_state(self, batch_size: int, dtype: mx.Dtype = mx.float32) -> MLSTMState:
        H, dh = self.num_heads, self.head_dim
        C = mx.zeros((batch_size, H, dh, dh), dtype=dtype)
        n = mx.zeros((batch_size, H, dh), dtype=dtype)
        m = mx.full((batch_size, H), -mx.inf, dtype=dtype)
        return (C, n, m)

    def __call__(self, x: mx.array) -> mx.array:
        B, T, _ = x.shape
        H, dh = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3) * self.scale
        v = self.v_proj(x).reshape(B, T, H, dh).transpose(0, 2, 1, 3)

        i_tilde = self.i_gate(x).transpose(0, 2, 1)
        f_tilde = self.f_gate(x).transpose(0, 2, 1)
        o = mx.sigmoid(self.o_gate(x))

        log_f = -nn.softplus(-f_tilde)
        idx = mx.arange(T)
        causal_mask = idx[:, None] >= idx[None, :]

        cum_log_f = mx.cumsum(log_f, axis=-1)
        log_f_matrix = cum_log_f[:, :, :, None] - cum_log_f[:, :, None, :]
        log_f_matrix = mx.where(causal_mask[None, None, :, :], log_f_matrix, -mx.inf)

        log_D = log_f_matrix + i_tilde[:, :, None, :]
        m = mx.max(log_D, axis=-1, keepdims=True)
        D_tilde = mx.exp(log_D - m)

        qk = q @ k.transpose(0, 1, 3, 2)
        C = qk * D_tilde

        n = mx.abs(mx.sum(C, axis=-1, keepdims=True))
        denom = mx.maximum(n, mx.exp(-m)) + 1e-6

        h = ((C / denom) @ v).transpose(0, 2, 1, 3)
        return o * self._norm_heads(h)

    def step(
        self,
        x_t: mx.array,
        state: MLSTMState | None = None,
    ) -> tuple[mx.array, MLSTMState]:
        B, _ = x_t.shape
        H, dh = self.num_heads, self.head_dim

        C_prev, n_prev, m_prev = self.init_state(B, x_t.dtype) if state is None else state

        q_t = self.q_proj(x_t).reshape(B, H, dh)
        k_t = self.k_proj(x_t).reshape(B, H, dh) * self.scale
        v_t = self.v_proj(x_t).reshape(B, H, dh)

        i_tilde = self.i_gate(x_t)
        f_tilde = self.f_gate(x_t)
        o_t = mx.sigmoid(self.o_gate(x_t))

        log_f = -nn.softplus(-f_tilde)
        m_t = mx.maximum(log_f + m_prev, i_tilde)
        f_t = mx.where(mx.isinf(m_prev), 0.0, mx.exp(log_f + m_prev - m_t))
        i_t = mx.exp(i_tilde - m_t)

        outer_vk = v_t[:, :, :, None] * k_t[:, :, None, :]
        C_t = f_t[:, :, None, None] * C_prev + i_t[:, :, None, None] * outer_vk
        n_t = f_t[:, :, None] * n_prev + i_t[:, :, None] * k_t

        num = (C_t @ q_t[:, :, :, None]).squeeze(-1)
        nq = mx.abs(mx.sum(n_t * q_t, axis=-1, keepdims=True))
        denom = mx.maximum(nq, mx.exp(-m_t)[:, :, None]) + 1e-6
        h_t = num / denom

        out = o_t * self._norm_heads(h_t)
        return out, (C_t, n_t, m_t)


class FlowRSSM(nn.Module):
    """RSSM whose prior is a conditional normalizing flow p(z_t | h_t) with mLSTM memory.

    The posterior is either
      * a conditional flow q(z_t | h_t, embed) over a fixed 128-d embedding
        (``flow_posterior=True``, the original behaviour; ``observe_step``), or
      * a Gaussian q(z_t | h_t, obs) produced by a Gaussian encoder head
        (``flow_posterior=False``; ``observe_step_gaussian`` and ``gaussian_kl``).
    """

    def __init__(
        self,
        action_dim: int,
        num_heads: int = 4,
        head_dim: int = 64,
        latent_dim: int = 32,
        flow_posterior: bool = True,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = num_heads * head_dim
        self.latent_dim = latent_dim

        self.in_proj = nn.Linear(latent_dim + action_dim, self.hidden_dim)
        self.memory = mLSTMCell(num_heads=num_heads, head_dim=head_dim)
        self.prior_flow = ConditionalNormalizingFlow(latent_dim, self.hidden_dim)
        self.posterior_flow = (
            ConditionalNormalizingFlow(latent_dim, self.hidden_dim + 128)
            if flow_posterior
            else None
        )

    def init_state(self, batch_size: int, dtype: mx.Dtype = mx.float32) -> MLSTMState:
        """Return initial zeroed mLSTM matrix memory state (C_0, n_0, m_0)."""
        return self.memory.init_state(batch_size, dtype=dtype)

    def observe_step(
        self,
        prev_latent: mx.array,
        prev_action: mx.array,
        state: MLSTMState | None,
        embed: mx.array,
    ) -> tuple[mx.array, MLSTMState, mx.array]:
        """Posterior step with a conditional normalizing flow.

        Returns ``(hidden, new_state, latent)``.
        """
        if self.posterior_flow is None:
            raise RuntimeError("flow_posterior=False: use observe_step_gaussian instead.")
        x = self.in_proj(mx.concatenate([prev_latent, prev_action], axis=-1))
        hidden, new_state = self.memory.step(x, state)
        post_context = mx.concatenate([hidden, embed], axis=-1)
        latent = self.posterior_flow.sample(post_context)
        return hidden, new_state, latent

    def observe_step_gaussian(
        self,
        prev_latent: mx.array,
        prev_action: mx.array,
        state: MLSTMState | None,
        features: mx.array,
        encoder: PosteriorEncoder,
        sample: bool = True,
    ) -> tuple[mx.array, MLSTMState, mx.array, mx.array, mx.array]:
        """Posterior step with a Gaussian encoder.

        ``encoder`` is a ``DreamerV3Encoder`` built with ``context_dim=hidden_dim`` and
        ``latent_dim=self.latent_dim``; ``features`` are its backbone features for this
        step. The posterior is conditioned on the *updated* ``hidden``, as in ``observe_step``.
        Returns ``(hidden, new_state, latent, mean, std)``. With ``sample=False`` the latent is the mean.
        """
        x = self.in_proj(mx.concatenate([prev_latent, prev_action], axis=-1))
        hidden, new_state = self.memory.step(x, state)
        mean, std = encoder.posterior(features, context=hidden)
        latent = gaussian_sample(mean, std) if sample else mean
        return hidden, new_state, latent, mean, std

    def gaussian_kl(
        self,
        mean: mx.array,
        std: mx.array,
        latent: mx.array,
        hidden: mx.array,
    ) -> mx.array:
        """Single-sample estimate of KL( N(mean, std^2) || prior_flow(. | hidden) ).

        KL = E_q[log q(z)] - E_q[log p(z)] = -H[q] - E_q[log p(z | h)].
        The entropy term is analytic; the flow term uses the reparameterized sample
        ``latent`` drawn from the same (mean, std), so gradients reach the encoder
        (through mean, std and the sample) and the flow prior. Shape: (*lead,).
        """
        return -gaussian_entropy(std) - self.prior_flow.log_prob(latent, hidden)

    def imagine_step(
        self,
        prev_latent: mx.array,
        prev_action: mx.array,
        state: MLSTMState | None,
    ) -> tuple[mx.array, MLSTMState, mx.array]:
        """Prior rollout step for imagination.

        Returns ``(hidden, new_state, latent)``.
        """
        x = self.in_proj(mx.concatenate([prev_latent, prev_action], axis=-1))
        hidden, new_state = self.memory.step(x, state)
        latent = self.prior_flow.sample(hidden)
        return hidden, new_state, latent


class ContinuousDreamerWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, latent_dim=32, num_bins=255):
        super().__init__()
        self.rssm = FlowRSSM(action_dim, hidden_dim, latent_dim)
        feature_dim = hidden_dim + latent_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ELU(), nn.Linear(128, 128))
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, obs_dim))
        self.reward_predictor = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, num_bins))
        self.continue_predictor = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, 1))

    def calculate_loss(self, obs, actions, rewards, continues):
        B, T, _ = obs.shape
        rewards = mx.squeeze(rewards, axis=-1)
        continues = mx.squeeze(continues, axis=-1)
        embeds = self.encoder(obs)
        
        hidden = mx.zeros((B, self.rssm.hidden_dim))
        latent = mx.zeros((B, self.rssm.latent_dim))
        features = []
        flow_kl_loss = 0.0

        for t in range(T):
            prev_act = actions[:, t-1] if t > 0 else mx.zeros_like(actions[:, 0])
            post_context = mx.concatenate([hidden, embeds[:, t]], axis=-1)
            prior_context = hidden
            
            hidden, latent = self.rssm.observe_step(latent, prev_act, hidden, embeds[:, t])
            log_prob_post = self.rssm.posterior_flow.log_prob(latent, post_context)
            log_prob_prior = self.rssm.prior_flow.log_prob(latent, prior_context)
            
            flow_kl_loss += mx.mean(log_prob_post - log_prob_prior)
            features.append(mx.concatenate([hidden, latent], axis=-1))

        features = mx.stack(features, axis=1)
        flow_kl_loss /= T

        pred_obs = self.decoder(features)
        recon_loss = mx.mean((pred_obs - obs) ** 2)

        pred_reward_logits = self.reward_predictor(features)
        reward_dist = TwoHotDist(pred_reward_logits)
        reward_loss = mx.mean((reward_dist.mean() - symlog(rewards)) ** 2)

        pred_cont_logits = self.continue_predictor(features)
        pred_cont_logits = mx.squeeze(pred_cont_logits, axis=-1)
        continue_loss = mx.mean(nn.losses.binary_cross_entropy(pred_cont_logits, continues))

        flow_kl_loss = mx.maximum(flow_kl_loss, 0.5)
        return recon_loss + reward_loss + continue_loss + 0.1 * flow_kl_loss

if __name__ == '__main__':
    batch_size, seq_len, num_heads, head_dim = 8, 64, 4, 32
    d_model = num_heads * head_dim

    model = mLSTMCell(num_heads=num_heads, head_dim=head_dim)
    x = mx.random.normal((batch_size, seq_len, d_model))

    # 1. Parallel forward pass
    y_parallel = model(x)

    # 2. Recurrent step-by-step pass (matches y_parallel within float precision)
    state = None
    step_outputs = []
    for t in range(seq_len):
        y_t, state = model.step(x[:, t, :], state)
        step_outputs.append(y_t)
    y_recurrent = mx.stack(step_outputs, axis=1)

    mx.eval(y_parallel, y_recurrent)
    print("Max diff:", mx.max(mx.abs(y_parallel - y_recurrent)).item())
