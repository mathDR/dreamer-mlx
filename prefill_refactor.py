import numpy as np

from replay_buffer import SequenceReplayBuffer

EPISODE_LEN = 200
obs_dim = int(np.prod(env.observation_space.shape))

# --- Buffer construction (replaces the old replay buffer class) ---------------------
replay_buffer = SequenceReplayBuffer(
    capacity=BUFFER_CAPACITY,  # e.g. 100_000 steps
    example={
        "obs": np.zeros(obs_dim, np.float32),
        "action": np.zeros(action_dim, np.float32),
        "reward": np.zeros(1, np.float32),
        "cont": np.zeros(1, np.float32),
        "is_last": np.bool_(False),  # marks the final step of each episode
    },
    terminal_key="is_last",  # sampled sequences never cross an episode boundary
    seed=SEED,
)

# --- Pre-fill with bang-bang exploration ---------------------------------------------
for _ in range(PRE_FILL_EPISODES):
    obs, _ = env.reset()

    # Bang-bang holding variables
    current_torque = np.random.choice([-1.0, 1.0])
    step_countdown = np.random.randint(10, 25)

    for t in range(EPISODE_LEN):
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
                "is_last": done or t == EPISODE_LEN - 1,
            }
        )

        obs = next_obs
        if done:
            break

# --- Training side: sample (B, L, ...) sequences ------------------------------------
# batch = replay_buffer.sample_sequences(BATCH_SIZE, SEQ_LEN)
# obs, act, rew, cont = batch["obs"], batch["action"], batch["reward"], batch["cont"]
# obs: (B, L, obs_dim), act: (B, L, action_dim), rew/cont: (B, L, 1)
