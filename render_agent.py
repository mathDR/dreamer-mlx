import gymnasium as gym
from gymnasium.utils.save_video import save_video
import mlx.core as mx
import numpy as np
import os

# Import your trained architectures from your main script
from main import ContinuousActor, ContinuousDreamerWorldModel

def render_learned_policy(episode_steps=200, video_folder="./videos"):
    print("🎬 Initializing Render Environment for Pendulum-v1...")
    env = gym.make("Pendulum-v1", render_mode="rgb_array_list")
    obs_dim = env.observation_space.shape[0]     # Extracts 3
    action_dim = env.action_space.shape[0]       # Extracts 1
    
    deter_dim, latent_dim = 256, 32
    feat_dim = deter_dim + latent_dim
    
    # 1. Instantiate the matching network shapes
    wm = ContinuousDreamerWorldModel(obs_dim=obs_dim, action_dim=action_dim, deter_dim=deter_dim, latent_dim=latent_dim)
    actor = ContinuousActor(feature_dim=feat_dim, action_dim=action_dim)
    
    # 2. Load the synchronized weights from disk
    print("📥 Loading saved parameter weights from Safetensors...")
    if os.path.exists("world_model.safetensors") and os.path.exists("actor.safetensors"):
        wm.load_weights("world_model.safetensors")
        actor.load_weights("actor.safetensors")
        print("✅ Weights loaded into memory successfully!")
    else:
        print("⚠️ Checkpoint files not found! Rendering with a random policy instead.")
    
    # 3. Reset environment and set up synchronized step-0 tracking variables
    obs, _ = env.reset()
    deter = mx.zeros((1, deter_dim))
    latent = mx.zeros((1, latent_dim))
    action_mx = mx.zeros((1, action_dim))
    
    total_reward = 0.0
    
    print("🕹️ Rolling out evaluation episode under learned policy...")
    for step in range(episode_steps):
        # Step 1: Update the continuous world model state trace with the current frame
        obs_mx = mx.reshape(mx.array(obs), (1, -1))
        embed = wm.encoder(obs_mx)
        deter, latent = wm.rssm.observe_step(latent, action_mx, deter, embed)
        
        # Step 2: Query policy deterministically from the fully synced hidden features
        features = mx.concatenate([deter, latent], axis=-1)
        action_mx, _, _ = actor.sample_and_log_prob(features)
        
        # Step 3: Scale torque action to match Gymnasium specs [-2.0, 2.0] and advance simulation
        action_np = np.array(action_mx).flatten() * 2.0
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        
        # Log the true unscaled reward feedback
        total_reward += float(reward)
        
        obs = next_obs
        if terminated or truncated:
            break

    print(f"📊 Episode Complete! Loaded Policy Evaluation True Return: {total_reward:.2f}")
    
    # 4. Unpack recorded visual frames
    recorded_frames = env.render()
    env.close()
    
    # 5. Build missing directory targets if needed
    if not os.path.exists(video_folder):
        os.makedirs(video_folder, exist_ok=True)
    
    # 6. Compile frames to an MP4 video file
    print(f"💾 Saving compiled video frames to '{video_folder}'...")
    save_video(
        frames=recorded_frames,
        video_folder=video_folder,
        fps=30,
        episode_index=0,
        name_prefix="trained_flow_dreamer"
    )
    print("🎉 Done! Video compiled successfully.")

if __name__ == "__main__":
    render_learned_policy()
