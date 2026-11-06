import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import gymnasium as gym
import math

# =====================================================================
# MATHEMATICAL STABILIZERS & CUSTOM UTILITIES
# =====================================================================

def symlog(x):
    return mx.sign(x) * mx.log(mx.abs(x) + 1.0)

def symexp(x):
    return mx.sign(x) * (mx.exp(mx.abs(x)) - 1.0)

class TwoHotDist:
    def __init__(self, logits, min_val=-20.0, max_val=20.0, num_bins=255):
        self.logits = logits
        self.bins = mx.linspace(min_val, max_val, num_bins)

    def mean(self):
        probs = mx.softmax(self.logits, axis=-1)
        return mx.sum(probs * self.bins, axis=-1)

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.xz_gates = nn.Linear(input_dim, 2 * hidden_dim)
        self.hz_gates = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.xn_gate = nn.Linear(input_dim, hidden_dim)
        self.hn_gate = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x, h):
        xz = self.xz_gates(x)
        hz = self.hz_gates(h)
        r_logits, z_logits = mx.split(xz + hz, 2, axis=-1)
        r = mx.sigmoid(r_logits)
        z = mx.sigmoid(z_logits)
        n = mx.tanh(self.xn_gate(x) + r * self.hn_gate(h))
        return (1.0 - z) * n + z * h

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

# =====================================================================
# ACTOR-CRITIC IMPLEMENTATIONS
# =====================================================================

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
    def __init__(self, feature_dim, num_bins=255, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, num_bins))
        self.bins = mx.linspace(-20.0, 20.0, num_bins)

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
        
        # Quantify relative advantages and normalize to stabilize step variance
        advantages = lambda_targets - imag_values_linear[:, :-1]
        advantages = (advantages - mx.mean(advantages)) / (mx.std(advantages) + 1e-6)
        
        # Actor Objective Loss Equation (Maximize Advantage + Entropy Regularization)
        actor_loss = mx.mean(-log_prob * mx.stop_gradient(advantages) - self.entropy_coef * entropy)

        # 6. Critic Objective Loss Equation
        # Project targets back into Symlog space to optimize against Two-Hot outputs
        symlog_targets = mx.stop_gradient(symlog(lambda_targets))
        
        # Map values back into cross-entropy indices matching the categorical output bins
        num_bins = critic_logits.shape[-1]
        target_indices = mx.clip((symlog_targets + 20.0) / (40.0 / (num_bins - 1)), 0, num_bins - 1)
        target_indices = mx.astype(target_indices, mx.int32)
        
        # --- FIX: Pure MLX alternative to one_hot() ---
        # 1. Generate an index range array matching your number of categories
        bin_indices = mx.arange(num_bins)  # Shape: (num_bins,)
        
        # 2. Add an explicit trailing dimension to target_indices to enable broadcasting
        # target_indices: [B, Horizon-1] -> [B, Horizon-1, 1]
        expanded_targets = mx.expand_dims(target_indices, axis=-1)
        
        # 3. Perform a broadcasted comparison against the bin indices range
        # Compares shape [B, Horizon-1, 1] == [num_bins] -> yields a [B, Horizon-1, num_bins] boolean array
        one_hot_targets = mx.astype(expanded_targets == bin_indices, mx.float32)
        
        # Compute the categorical cross entropy loss
        critic_loss = mx.mean(nn.losses.cross_entropy(critic_logits[:, :-1], one_hot_targets))

        return actor_loss, critic_loss


class SequenceReplayBuffer:
    """
    Stores full trajectories to sample chronological sub-sequences 
    matching Dreamer's sequence length requirements.
    """
    def __init__(self, capacity=500, seq_len=16):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = []

    def add_episode(self, observations, actions, rewards, continues):
        """Appends a completed episode trajectory dictionary to the buffer memory."""
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)  # Evict oldest episode if capacity is exceeded
            
        self.buffer.append({
            'obs': np.array(observations, dtype=np.float32),
            'actions': np.array(actions, dtype=np.float32),
            'rewards': np.array(rewards, dtype=np.float32),
            'continues': np.array(continues, dtype=np.float32)
        })

    def sample_batch(self, batch_size):
        """Samples a batch of synchronized, temporal sub-sequences."""
        bo, ba, br, bc = [], [], [], []
        
        while len(bo) < batch_size:
            # 1. Pick a random full episode from the history bank
            ep = self.buffer[np.random.randint(0, len(self.buffer))]
            ep_len = len(ep['obs'])
            
            # Skip episodes that are structurally too short to yield a valid sequence slice
            if ep_len <= self.seq_len:
                continue
                
            # 2. Pick a valid randomized start step window index
            start = np.random.randint(0, ep_len - self.seq_len)
            end = start + self.seq_len
            
            # 3. Harvest synchronized slices across all parameter dimensions
            bo.append(ep['obs'][start:end])
            ba.append(ep['actions'][start:end])
            br.append(ep['rewards'][start:end])
            bc.append(ep['continues'][start:end])
            
        # Compile lists into stacked MLX execution array structures
        return (
            mx.array(np.stack(bo)),
            mx.array(np.stack(ba)),
            mx.array(np.stack(br)),
            mx.array(np.stack(bc))
        )

    def __len__(self):
        """Exposes current total episode count inside the tracking array."""
        return len(self.buffer)

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    BATCH_SIZE, SEQ_LEN = 8, 16
    
    print("Initializing Farama Gymnasium Pendulum Environment...")
    env = gym.make("Pendulum-v1")
    
    # --- FIX: Extract the integer dimension from the shape tuple ---
    obs_dim = env.observation_space.shape[0]     # Unpacks (3,) -> 3
    action_dim = env.action_space.shape[0]       # Unpacks (1,) -> 1
    
    replay_buffer = SequenceReplayBuffer(capacity=200, seq_len=SEQ_LEN)
    wm = ContinuousDreamerWorldModel(obs_dim=obs_dim, action_dim=action_dim)
    
    feat_dim = wm.rssm.deter_dim + wm.rssm.latent_dim
    actor = ContinuousActor(feature_dim=feat_dim, action_dim=action_dim)
    critic = ContinuousCritic(feature_dim=feat_dim)
    ac_updater = ActorCriticUpdater(actor, critic, wm)

    wm_opt = optim.Adam(learning_rate=3e-4)
    actor_opt = optim.Adam(learning_rate=1e-4)
    critic_opt = optim.Adam(learning_rate=1e-4)

    # 1. State-dependent gradient function wrappers
    wm_grad_fn = nn.value_and_grad(wm, lambda m, o, a, r, c: m.calculate_loss(o, a, r, c))

    class ActorCriticContainer(nn.Module):
        def __init__(self, actor_mod, critic_mod):
            super().__init__()
            self.actor = actor_mod
            self.critic = critic_mod
            
    ac_container = ActorCriticContainer(actor, critic)
    
    def ac_loss_fn(m, sf):
        ac_updater.actor = m.actor
        ac_updater.critic = m.critic
        return ac_updater.calculate_losses(sf)
        
    ac_grad_fn = nn.value_and_grad(ac_container, ac_loss_fn)

    # 2. Pre-fill Buffer using a baseline uniform action trace
    print("Pre-filling Replay Buffer...")
    for _ in range(5):
        obs, _ = env.reset() 
        obs_seq = [np.array(obs, dtype=np.float32).flatten()]
        act_seq, rew_seq, cont_seq = [], [], []
        
        for _ in range(200):
            action = env.action_space.sample() 
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            obs_seq.append(np.array(next_obs, dtype=np.float32).flatten())
            act_seq.append(np.array(action, dtype=np.float32).flatten())
            rew_seq.append(np.array([reward], dtype=np.float32))
            cont_seq.append(np.array([0.0 if (terminated or truncated) else 1.0], dtype=np.float32))
            
        replay_buffer.add_episode(obs_seq[:200], act_seq[:200], rew_seq[:200], cont_seq[:200])

    # 3. Interactive Reinforcement Training Loop
    print("Beginning Continuous Flow Dreamer Execution Loop...")
    for episode in range(50):
        obs, _ = env.reset() 
        deter = mx.zeros((1, wm.rssm.deter_dim))
        latent = mx.zeros((1, wm.rssm.latent_dim))
        
        episode_reward = 0.0
        obs_seq = [np.array(obs, dtype=np.float32).flatten()]
        act_seq, rew_seq, cont_seq = [], [], []

        # --- ENVIRONMENT STEPPING INTERACTION PHASE ---
        for step in range(200):
            features = mx.concatenate([deter, latent], axis=-1)
            action_mx, _, _ = actor.sample_and_log_prob(features)
            
            # --- FIX: Flatten the numpy action vector completely to shape (1,) ---
            action_np = np.array(action_mx).flatten() * 2.0 
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            
            # --- FIX: Force reward to be a pure scalar float to avoid matrix morphing ---
            reward_scalar = float(reward)
            episode_reward += reward_scalar

            # Continuous model embedding step (Forces clean shape (1, 3))
            obs_mx = mx.reshape(mx.array(next_obs), (1, -1)) 
            embed = wm.encoder(obs_mx)
            deter, latent = wm.rssm.observe_step(latent, action_mx, deter, embed)

            obs_seq.append(np.array(next_obs, dtype=np.float32).flatten())
            act_seq.append(np.array(action_np / 2.0, dtype=np.float32).flatten())  
            rew_seq.append(np.array([reward_scalar], dtype=np.float32)) # Clean (1,) vector
            cont_seq.append(np.array([0.0 if (terminated or truncated) else 1.0], dtype=np.float32))

        replay_buffer.add_episode(obs_seq[:200], act_seq[:200], rew_seq[:200], cont_seq[:200])

        # --- GRAPH OPTIMIZATION PHASE ---
        b_obs, b_actions, b_rewards, b_continues = replay_buffer.sample_batch(BATCH_SIZE)
        
        # A. Optimize World Model
        wm_loss, wm_grads = wm_grad_fn(wm, b_obs, b_actions, b_rewards, b_continues)
        wm_opt.update(wm, wm_grads)

        # B. Extract features along true trajectories to anchor Dream Imagination
        embeds = wm.encoder(b_obs)
        d_s = mx.zeros((BATCH_SIZE, wm.rssm.deter_dim))
        l_s = mx.zeros((BATCH_SIZE, wm.rssm.latent_dim))
        sf_list = []
        for t in range(SEQ_LEN):
            prev_a = b_actions[:, t-1] if t > 0 else mx.zeros_like(b_actions[:, 0])
            d_s, l_s = wm.rssm.observe_step(l_s, prev_a, d_s, embeds[:, t])
            sf_list.append(mx.concatenate([d_s, l_s], axis=-1))
        
        start_features = mx.reshape(mx.stack(sf_list, axis=1), (-1, feat_dim))
        
        # C. Optimize Actor and Critic networks simultaneously
        (a_loss, c_loss), ac_grads = ac_grad_fn(ac_container, start_features)
        
        # Pull parameters directly from the returned structural gradients dictionary
        actor_opt.update(actor, ac_grads["actor"])
        critic_opt.update(critic, ac_grads["critic"])

        # D. Flush unified memory graphs to prevent RAM accumulation leaks
        mx.eval(wm.parameters(), actor.parameters(), critic.parameters(), 
                wm_opt.state, actor_opt.state, critic_opt.state)
                
        # --- FIX: Removed .item() from episode_reward since it is already a Python float ---
        print(f"Episode {episode:02d} | Return: {episode_reward:7.2f} | "
              f"WM Loss: {wm_loss.item():.3f} | Actor Loss: {a_loss.item():.3f} | Critic Loss: {c_loss.item():.3f}")


if __name__ == "__main__":
    main()

