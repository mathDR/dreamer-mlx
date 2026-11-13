import math

import mlx.core as mx
from mlx import nn

from dreamer_mlx.utils import TwoHotDist, symexp, symlog


class ContinuousActor(nn.Module):
    def __init__(self, feature_dim, action_dim, hidden_dim=128, min_std=1e-4, max_std=1.0):
        super().__init__()
        self.min_std, self.max_std = min_std, max_std
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU())
        self.mu_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def sample_and_log_prob(self, features):
        x = self.net(features)
        mu = self.mu_head(x)
        log_std = mx.tanh(self.std_head(x))
        std = self.min_std + (self.max_std - self.min_std) * 0.5 * (log_std + 1.0)
        
        eps = mx.random.normal(shape=mu.shape)
        raw_action = mu + std * eps
        action = mx.tanh(raw_action)
        
        log_prob_normal = -0.5 * (((raw_action - mu) / std) ** 2 + 2 * mx.log(std) + mx.log(2 * math.pi))
        tanh_correction = mx.sum(mx.log(1.0 - action ** 2 + 1e-6), axis=-1)
        log_prob = mx.sum(log_prob_normal, axis=-1) - tanh_correction
        entropy = mx.sum(mx.log(std) + 0.5 * (1.0 + mx.log(2 * math.pi)), axis=-1)
        return action, log_prob, entropy

class ContinuousCritic(nn.Module):
    def __init__(self, feature_dim: int, num_bins: int=255, hidden_dim: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(), 
            nn.Linear(hidden_dim, num_bins)
        )
        # FIX: Expand tracking boundaries to match environment returns
        self.bins = mx.linspace(-100.0, 100.0, num_bins) 

    def __call__(self, features):
        logits = self.net(features)
        probs = mx.softmax(logits, axis=-1)
        return mx.sum(probs * self.bins, axis=-1), logits

class ActorCriticUpdater:
    def __init__(self, actor, critic, world_model, gamma=0.99, lam=0.95, entropy_coef=1e-3):
        self.actor, self.critic, self.world_model = actor, critic, world_model
        self.gamma, self.lam, self.entropy_coef = gamma, lam, entropy_coef

    def rollout_imagination(self, start_features, horizon=12):
        deter_dim = self.world_model.rssm.deter_dim
        deter = start_features[:, :deter_dim]
        latent = start_features[:, deter_dim:]
        imag_features = []
        for _ in range(horizon):
            current_feat = mx.concatenate([deter, latent], axis=-1)
            action, _, _ = self.actor.sample_and_log_prob(current_feat)
            deter, latent = self.world_model.rssm.imagine_step(latent, action, deter)
            imag_features.append(mx.concatenate([deter, latent], axis=-1))
        return mx.stack(imag_features, axis=1)
    
    def calculate_losses(self, start_features):
        """
        Executes imagination rollouts and computes distinct total losses 
        for both the Actor policy optimization and Critic evaluations.
        """
        # 1. Generate Dream Horizon path [B * T, Horizon, FeatureDim]
        imag_features = self.rollout_imagination(start_features)
        
        # 2. Query World Model predictors over the imagined trajectory
        pred_reward_logits = self.world_model.reward_predictor(imag_features)
        pred_cont_logits = self.world_model.continue_predictor(imag_features)
        
        # Decode rewards and continues using their matching stabilizers
        imag_rewards = symexp(TwoHotDist(pred_reward_logits).mean())
        imag_continues = mx.sigmoid(pred_cont_logits)[..., 0]
        
        # 3. Query Critic values across trajectory states and convert back to linear scale
        imag_values, critic_logits = self.critic(imag_features)
        imag_values_linear = symexp(imag_values)

        # 4. Construct Generalized Advantage Targets
        # Split data slices to align matching indices cleanly up to (Horizon - 1)
        rewards = imag_rewards[:, :-1]
        continues = imag_continues[:, :-1]
        
        T_steps = rewards.shape[1]
        targets = [mx.zeros_like(imag_values_linear[:, -1])] * T_steps
        next_target = imag_values_linear[:, -1]
        
        # Work backward recursively from the horizon edge
        for t in reversed(range(T_steps)):
            targets[t] = rewards[:, t] + continues[:, t] * self.gamma * (
                (1.0 - self.lam) * imag_values_linear[:, t + 1] + self.lam * next_target
            )
            next_target = targets[t]
        lambda_targets = mx.stack(targets, axis=1)
        
        # 5. Evaluate policy actions matching target parameters
        _, log_prob, entropy = self.actor.sample_and_log_prob(imag_features[:, :-1])
        
        # --- ALIGNMENT FIX: Compare targets against current values (index 1 to end) ---
        # Lambda targets map out future values, so they must be compared with the 
        # model's forward step evaluations to compute true advantages.
        baseline_values = imag_values_linear[:, 1:] 
        advantages = lambda_targets - baseline_values
        
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-6)
        actor_loss = mx.mean(-log_prob * mx.stop_gradient(advantages) - self.entropy_coef * entropy)

        # 6. Critic Objective Loss Equation
        # Project targets back into Symlog space to optimize against Two-Hot outputs
        symlog_targets = mx.stop_gradient(symlog(lambda_targets))
        
        # FIX: Update the index mapping arithmetic to match our new [-100, 100] bounds
        num_bins = critic_logits.shape[-1]
        target_indices = mx.clip((symlog_targets + 100.0) / (200.0 / (num_bins - 1)), 0, num_bins - 1)
        target_indices = mx.astype(target_indices, mx.int32)
        
        bin_indices = mx.arange(num_bins)
        expanded_targets = mx.expand_dims(target_indices, axis=-1)
        one_hot_targets = mx.astype(expanded_targets == bin_indices, mx.float32)
        
        critic_loss = mx.mean(nn.losses.cross_entropy(critic_logits[:, :-1], one_hot_targets))

        return actor_loss, critic_loss
