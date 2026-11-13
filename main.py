

import gymnasium as gym
import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from dreamer_mlx.policies import (
    ActorCriticUpdater,
    ContinuousActor,
    ContinuousCritic,
)
from mlx import nn
from mlx.utils import tree_map

from dreamer_mlx.world_model import ContinuousDreamerWorldModel


class OUExplorationNoise:
    """Generates temporally correlated random walks to build mechanical energy."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state


# =====================================================================
# ACTOR-CRITIC IMPLEMENTATIONS
# =====================================================================




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

# =====================================================================
# MAIN EXECUTION PIPELINE
# =====================================================================

def main():
    BATCH_SIZE, SEQ_LEN = 16, 64
    PRE_FILL_EPISODES = 30
    MAX_EPISODES = 500
    
    print("Initializing Farama Gymnasium Pendulum Environment...")
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]     # Unpacks (3,) -> 3
    action_dim = env.action_space.shape[0]       # Unpacks (1,) -> 1
    
    replay_buffer = SequenceReplayBuffer(capacity=500, seq_len=SEQ_LEN)
    wm = ContinuousDreamerWorldModel(obs_dim=obs_dim, action_dim=action_dim, deter_dim=256, latent_dim=32)
    feat_dim = wm.rssm.deter_dim + wm.rssm.latent_dim
    actor = ContinuousActor(feature_dim=feat_dim, action_dim=action_dim)
    critic = ContinuousCritic(feature_dim=feat_dim)
    ac_updater = ActorCriticUpdater(actor, critic, wm, gamma=0.99, lam=0.95, entropy_coef=1e-2)

    # Wrap actor and critic into a container for multi-module gradient tracking
    class ActorCriticContainer(nn.Module):
        def __init__(self, actor_mod, critic_mod):
            super().__init__()
            self.actor = actor_mod
            self.critic = critic_mod
            
    ac_container = ActorCriticContainer(actor, critic)

    wm_opt = optim.Adam(learning_rate=3e-4)
    actor_opt = optim.Adam(learning_rate=3e-5)
    critic_opt = optim.Adam(learning_rate=1e-3)

    # State-dependent gradient function wrappers
    wm_grad_fn = nn.value_and_grad(wm, lambda m, o, a, r, c: m.calculate_loss(o, a, r, c))
    
    def ac_loss_fn(m, sf):
        ac_updater.actor = m.actor
        ac_updater.critic = m.critic
        return ac_updater.calculate_losses(sf)
        
    ac_grad_fn = nn.value_and_grad(ac_container, ac_loss_fn)

    # 1. PRE-FILL PHASE: Gather random trajectories to warm up the buffer
    print(f"Pre-filling Replay Buffer with {PRE_FILL_EPISODES} episodes...")
    ou_noise = OUExplorationNoise(size=env.action_space.shape)

    # --- INSIDE YOUR main() FUNCTION (SWAP THE PRE-FILL LOOP) ---
    print(f"Pre-filling Replay Buffer with {PRE_FILL_EPISODES} high-energy explosive episodes...")
    
    for _ in range(PRE_FILL_EPISODES):
        obs, _ = env.reset() 
        obs_seq = [np.array(obs, dtype=np.float32).flatten()]
        act_seq, rew_seq, cont_seq = [], [], []
        
        deter = mx.zeros((1, wm.rssm.deter_dim))
        latent = mx.zeros((1, wm.rssm.latent_dim))
        action_mx = mx.zeros((1, action_dim))
        
        # Bang-Bang holding variables
        current_torque = np.random.choice([-1.0, 1.0])
        hold_duration = np.random.randint(10, 25)
        step_countdown = hold_duration
        
        for _ in range(200):
            obs_mx = mx.reshape(mx.array(obs), (1, -1))
            embed = wm.encoder(obs_mx)
            deter, latent = wm.rssm.observe_step(latent, action_mx, deter, embed)
            
            # Manage holding timer
            step_countdown -= 1
            if step_countdown <= 0:
                current_torque = -current_torque # Reverse direction completely
                hold_duration = np.random.randint(10, 25)
                step_countdown = hold_duration
                
            # Scale up to max environmental limits [-2.0, 2.0]
            action = np.array([current_torque], dtype=np.float32) * 2.0
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            reward_scalar = float(reward)
            
            obs_seq.append(np.array(next_obs, dtype=np.float32).flatten())
            act_seq.append(np.array(action, dtype=np.float32).flatten())
            rew_seq.append(np.array([reward_scalar * 0.05], dtype=np.float32))
            cont_seq.append(np.array([0.0 if (terminated or truncated) else 1.0], dtype=np.float32))
            
            obs = next_obs
            action_mx = mx.array(action)[None, ...]
            
        replay_buffer.add_episode(obs_seq[:200], act_seq[:200], rew_seq[:200], cont_seq[:200])

    # 2. INTERACTIVE ONLINE REINFORCEMENT TRAINING LOOP
    print("Beginning Continuous Flow Dreamer Execution Loop...")
    for episode in range(MAX_EPISODES):
        obs, _ = env.reset() 
        deter = mx.zeros((1, wm.rssm.deter_dim))
        latent = mx.zeros((1, wm.rssm.latent_dim))
        action_mx = mx.zeros((1, action_dim))
        
        episode_reward = 0.0
        obs_seq = [np.array(obs, dtype=np.float32).flatten()]
        act_seq, rew_seq, cont_seq = [], [], []

        for step in range(200):
            obs_mx = mx.reshape(mx.array(obs), (1, -1)) 
            embed = wm.encoder(obs_mx)
            deter, latent = wm.rssm.observe_step(latent, action_mx, deter, embed)

            features = mx.concatenate([deter, latent], axis=-1)
            action_mx, _, _ = actor.sample_and_log_prob(features)
            
            noise = np.random.normal(0, 0.4, size=env.action_space.shape) # Injected variance
            action_np = np.clip(np.array(action_mx).flatten() + noise, -1.0, 1.0) * 2.0 
            
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            reward_scalar = float(reward)
            episode_reward += reward_scalar 

            # --- SHAPING FIX: Add an explicit height bonus to reward_scalar ---
            # next_obs[0] is cos(theta). It is +1.0 at the absolute top, -1.0 at the bottom.
            # This smoothly guides the model to favor height positions.
            height_bonus = float(next_obs[0]) * 5.0
            shaped_reward = (reward_scalar + height_bonus) * 0.05

            obs_seq.append(np.array(next_obs, dtype=np.float32).flatten())
            act_seq.append(np.array(action_np / 2.0, dtype=np.float32).flatten())  
            rew_seq.append(np.array([shaped_reward], dtype=np.float32)) # Fed to the World Model
            cont_seq.append(np.array([0.0 if (terminated or truncated) else 1.0], dtype=np.float32))
            
            obs = next_obs


        replay_buffer.add_episode(obs_seq[:200], act_seq[:200], rew_seq[:200], cont_seq[:200])

        # --- GRAPH OPTIMIZATION PHASE ---
        b_obs, b_actions, b_rewards, b_continues = replay_buffer.sample_batch(BATCH_SIZE)
        
        # A. Optimize World Model Graph
        wm_loss, wm_grads = wm_grad_fn(wm, b_obs, b_actions, b_rewards, b_continues)
        wm_opt.update(wm, wm_grads)
        a_loss_val, c_loss_val = 0.0, 0.0

        if episode >= 50:
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
            
            actor_grads = tree_map(lambda x: mx.clip(x, -10.0, 10.0), ac_grads["actor"])
            critic_grads = tree_map(lambda x: mx.clip(x, -10.0, 10.0), ac_grads["critic"])
            
            actor_opt.update(actor, actor_grads)
            critic_opt.update(critic, critic_grads)
            
            a_loss_val, c_loss_val = a_loss.item(), c_loss.item()

        # D. Flush unified memory graphs to clear evaluated traces
        mx.eval(wm.parameters(), actor.parameters(), critic.parameters(), 
                wm_opt.state, actor_opt.state, critic_opt.state)
                
        print(f"Episode {episode:03d} | Return: {episode_reward:7.2f} | "
              f"WM Loss: {wm_loss.item():.3f} | Actor Loss: {a_loss_val:.3f} | Critic Loss: {c_loss_val:.3f}")
    
    # --- SAVE WEIGHTS TO DISK ---
    print("\n💾 Saving trained model checkpoints...")
    wm.save_weights("world_model.safetensors")
    actor.save_weights("actor.safetensors")
    critic.save_weights("critic.safetensors")
    print("🎉 All weights saved successfully to disk as Safetensors!")

if __name__ == "__main__":
    main()
