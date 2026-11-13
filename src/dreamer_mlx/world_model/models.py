import mlx.core as mx
from mlx import nn

from dreamer_mlx.utils import TwoHotDist, symlog


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

class FlowRSSM(nn.Module):
    def __init__(self, action_dim, deter_dim=256, latent_dim=32):
        super().__init__()
        self.deter_dim = deter_dim
        self.latent_dim = latent_dim
        self.cell = GRUCell(input_dim=latent_dim + action_dim, hidden_dim=deter_dim)
        self.prior_flow = ConditionalNormalizingFlow(latent_dim, deter_dim)
        self.posterior_flow = ConditionalNormalizingFlow(latent_dim, deter_dim + 128)

    def observe_step(self, prev_latent, prev_action, deter, embed):
        x = mx.concatenate([prev_latent, prev_action], axis=-1)
        deter = self.cell(x, deter)
        post_context = mx.concatenate([deter, embed], axis=-1)
        latent = self.posterior_flow.sample(post_context)
        return deter, latent

    def imagine_step(self, prev_latent, prev_action, deter):
        x = mx.concatenate([prev_latent, prev_action], axis=-1)
        deter = self.cell(x, deter)
        latent = self.prior_flow.sample(deter)
        return deter, latent

class ContinuousDreamerWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, deter_dim=256, latent_dim=32, num_bins=255):
        super().__init__()
        self.rssm = FlowRSSM(action_dim, deter_dim, latent_dim)
        feature_dim = deter_dim + latent_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.ELU(), nn.Linear(128, 128))
        self.decoder = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, obs_dim))
        self.reward_predictor = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, num_bins))
        self.continue_predictor = nn.Sequential(nn.Linear(feature_dim, 128), nn.ELU(), nn.Linear(128, 1))

    def calculate_loss(self, obs, actions, rewards, continues):
        B, T, _ = obs.shape
        rewards = mx.squeeze(rewards, axis=-1)
        continues = mx.squeeze(continues, axis=-1)
        embeds = self.encoder(obs)
        
        deter = mx.zeros((B, self.rssm.deter_dim))
        latent = mx.zeros((B, self.rssm.latent_dim))
        features = []
        flow_kl_loss = 0.0

        for t in range(T):
            prev_act = actions[:, t-1] if t > 0 else mx.zeros_like(actions[:, 0])
            post_context = mx.concatenate([deter, embeds[:, t]], axis=-1)
            prior_context = deter
            
            deter, latent = self.rssm.observe_step(latent, prev_act, deter, embeds[:, t])
            log_prob_post = self.rssm.posterior_flow.log_prob(latent, post_context)
            log_prob_prior = self.rssm.prior_flow.log_prob(latent, prior_context)
            
            flow_kl_loss += mx.mean(log_prob_post - log_prob_prior)
            features.append(mx.concatenate([deter, latent], axis=-1))

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
