

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
from dreamer_mlx.data import SequenceReplayBuffer

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


def main():
    BATCH_SIZE, SEQ_LEN = 16, 64
    PRE_FILL_EPISODES = 30
    MAX_EPISODES = 500

    EPISODE_LEN = 200
    
    print("Initializing Farama Gymnasium Pendulum Environment...")
    env = gym.make("Pendulum-v1")
    obs_dim = env.observation_space.shape[0]     # Unpacks (3,) -> 3
    action_dim = env.action_space.shape[0]       # Unpacks (1,) -> 1
    
    replay_buffer = SequenceReplayBuffer(
        capacity=100_000,  # e.g. 100_000 steps
        example={
            "obs": np.zeros(obs_dim, np.float32),
            "action": np.zeros(action_dim, np.float32),
            "reward": np.zeros(1, np.float32),
            "cont": np.zeros(1, np.float32),
            "is_last": np.bool_(False),  # marks the final step of each episode
        },
        terminal_key="is_last",  # sampled sequences never cross an episode boundary
        seed=4,
    )
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

        # Bang-bang holding variables
        current_torque = np.random.choice([-1.0, 1.0])
        step_countdown = np.random.randint(10, 25)

        for episode in range(EPISODE_LEN):
            # Manage holding timer
            step_countdown -= 1
            if step_countdown <= 0:
                current_torque = -current_torque  # reverse direction completely
                step_countdown = np.random.randint(10, 25)

            # Scale up to max environmental limits [-2.0, 2.0]
            action = np.array([current_torque], dtype=np.float32) * 2.0
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # obs_t pairs with action_t, reward_t, cont_t, exactly as before
            replay_buffer.add(
                {
                    "obs": np.asarray(obs, dtype=np.float32).ravel(),
                    "action": action,
                    "reward": np.array([float(reward) * 0.05], dtype=np.float32),
                    "cont": np.array([0.0 if done else 1.0], dtype=np.float32),
                    "is_last": done or episode == EPISODE_LEN - 1,
                }
            )

            obs = next_obs
            if done:
                break

        # --- GRAPH OPTIMIZATION PHASE ---
        b = replay_buffer.sample_sequences(32, 16)
        b_obs, b_actions, b_rewards, b_continues = b["obs"], b["action"], b["reward"], b["cont"]
        
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
                breakpoint()
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
