"""Fill a SequenceReplayBuffer with pixel observations from dm_control's acrobot.

Rendering backend: on macOS the default (GLFW) works. On headless Linux set
``MUJOCO_GL=egl`` (GPU) or ``MUJOCO_GL=osmesa`` (CPU) *before* importing dm_control.

Memory: one 64x64x3 uint8 frame is 12 KB, so 100k steps is ~1.2 GB.
"""

import numpy as np
from dm_control import suite

from dreamer_mlx.data import SequenceReplayBuffer

IMAGE_SIZE = 64
ACTION_REPEAT = 2  # env steps per agent step (acrobot's time limit is 1000 env steps)
CAMERA_ID = 0
REWARD_SCALE = 1.0  # acrobot rewards are already in [0, 1]
PRE_FILL_EPISODES = 20
SEED = 0


class BangBangPolicy:
    """Hold min/max action for a random duration, then flip. Works for any action_spec."""

    def __init__(self, action_spec, hold_range=(10, 25), rng=None):
        self.low = np.asarray(action_spec.minimum, dtype=np.float32)
        self.high = np.asarray(action_spec.maximum, dtype=np.float32)
        self.hold_range = hold_range  # in agent steps; high end exclusive
        self.rng = rng if rng is not None else np.random.default_rng()
        self.reset()

    def reset(self):
        self.sign = self.rng.choice([-1.0, 1.0])
        self.countdown = int(self.rng.integers(*self.hold_range))

    def __call__(self) -> np.ndarray:
        self.countdown -= 1
        if self.countdown <= 0:
            self.sign = -self.sign  # reverse direction completely
            self.countdown = int(self.rng.integers(*self.hold_range))
        return (self.high if self.sign > 0 else self.low).copy()


def make_env(domain: str = "acrobot", task: str = "swingup", seed: int = SEED):
    return suite.load(domain, task, task_kwargs={"random": seed})


def render(env, size: int = IMAGE_SIZE, camera_id: int = CAMERA_ID) -> np.ndarray:
    """Render an (size, size, 3) uint8 RGB frame."""
    return env.physics.render(height=size, width=size, camera_id=camera_id)


def make_buffer(env, capacity: int, size: int = IMAGE_SIZE, seed: int = SEED):
    action_dim = int(np.prod(env.action_spec().shape))
    return SequenceReplayBuffer(
        capacity=capacity,
        example={
            "image": np.zeros((size, size, 3), np.uint8),
            "action": np.zeros(action_dim, np.float32),
            "reward": np.zeros(1, np.float32),
            "cont": np.zeros(1, np.float32),
            "is_last": np.bool_(False),
        },
        terminal_key="is_last",  # sequences never cross an episode boundary
        seed=seed,
    )


def collect_episode(
    env,
    buffer: SequenceReplayBuffer,
    policy,
    *,
    size: int = IMAGE_SIZE,
    camera_id: int = CAMERA_ID,
    action_repeat: int = ACTION_REPEAT,
    reward_scale: float = REWARD_SCALE,
):
    """Run one episode, adding every agent step to the buffer.

    Each stored step pairs the frame *before* acting with the action taken, the reward
    accumulated over the action repeat, and the continuation flag that follows.
    Returns ``(episode_return, num_agent_steps)``.
    """
    timestep = env.reset()
    policy.reset()
    frame = render(env, size, camera_id)
    episode_return, n_steps = 0.0, 0

    while not timestep.last():
        action = policy()

        reward = 0.0
        for _ in range(action_repeat):
            timestep = env.step(action)
            reward += timestep.reward or 0.0  # reward is None only on the reset step
            if timestep.last():
                break

        buffer.add(
            {
                "image": frame,
                "action": action,
                "reward": np.array([reward * reward_scale], dtype=np.float32),
                # dm_control's discount stays 1.0 at a time-limit end (truncation) and is
                # 0.0 only on true termination, so cont is the correct bootstrap flag.
                "cont": np.array([timestep.discount], dtype=np.float32),
                "is_last": timestep.last(),  # episode boundary, independent of cont
            }
        )

        frame = render(env, size, camera_id)
        episode_return += reward
        n_steps += 1

    return episode_return, n_steps


def prefill(env, buffer, policy, num_episodes: int = PRE_FILL_EPISODES, **kwargs):
    for i in range(num_episodes):
        ret, n = collect_episode(env, buffer, policy, **kwargs)
        print(f"episode {i + 1}/{num_episodes}: {n} steps, return {ret:.2f}")


if __name__ == "__main__":
    env = make_env()
    steps_per_episode = int(env._step_limit) // ACTION_REPEAT  # env steps / action repeat
    buffer = make_buffer(env, capacity=PRE_FILL_EPISODES * steps_per_episode)
    policy = BangBangPolicy(env.action_spec(), rng=np.random.default_rng(SEED))

    prefill(env, buffer, policy)
    print(f"buffer: {len(buffer)} steps")

    batch = buffer.sample_sequences(batch_size=8, seq_len=16)
    print({k: (v.shape, v.dtype) for k, v in batch.items()})

    # Feeding an MLX conv encoder (MLX convs are NHWC, matching the stored layout):
    #   import mlx.core as mx
    #   img = batch["image"].astype(mx.float32) / 255.0 - 0.5     # (B, L, H, W, 3)
    #   B, L = img.shape[:2]
    #   embed = encoder(img.reshape(B * L, *img.shape[2:])).reshape(B, L, -1)
