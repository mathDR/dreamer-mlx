import mlx.core as mx
import numpy as np
import pytest
from mlx import nn

from main import (
    ActorCriticUpdater,
    ConditionalNormalizingFlow,
    ContinuousActor,
    ContinuousCritic,
    ContinuousDreamerWorldModel,
    GRUCell,
    SequenceReplayBuffer,
    TwoHotDist,
    symexp,
    symlog,
)

# =====================================================================
# FIXTURES (Manages structural configuration settings across tests)
# =====================================================================

@pytest.fixture
def config():
    """Provides standard dimensional bounds matching the Pendulum-v1 setup."""
    return {
        "B": 4,           # Batch Size
        "T": 10,          # Sequence Length
        "obs_dim": 3,     # Observation vector width
        "action_dim": 1,  # Action vector width
        "deter_dim": 64,
        "latent_dim": 16,
        "feat_dim": 64 + 16, # deter_dim + latent_dim
    }

# =====================================================================
# 1. TEST MATHEMATICAL STABILIZERS
# =====================================================================

def test_symlog_symexp_inversion():
    """Assert that symlog and symexp are perfect mathematical inverses."""
    targets = mx.array([-100.0, -1.0, 0.0, 1.0, 100.0])
    compressed = symlog(targets)
    recovered = symexp(compressed)
    
    np.testing.assert_allclose(np.array(recovered), np.array(targets), rtol=1e-5)


def test_two_hot_distribution_mean(config):
    """Assert TwoHotDist properly decodes categorical bin logits into continuous expectations."""
    num_bins = 255
    mask = np.zeros((config["B"], num_bins), dtype=np.float32)
    mask[:, -1] = 100.0  # Force soft-max probability concentration to 1.0 at maximum bound (20.0)
    
    dist = TwoHotDist(mx.array(mask), min_val=-20.0, max_val=20.0, num_bins=num_bins)
    mean_val = dist.mean()
    
    assert mean_val.shape == (config["B"],)
    np.testing.assert_allclose(np.array(mean_val), 20.0, atol=1e-3)

# =====================================================================
# 2. TEST RECURRENT UNIT AND FLOW STRUCTURES
# =====================================================================

def test_custom_gru_cell_step(config):
    """Verify that the custom MLX GRUCell tracks states and maintains structural shapes."""
    cell = GRUCell(input_dim=config["latent_dim"] + config["action_dim"], hidden_dim=config["deter_dim"])
    x = mx.random.normal((config["B"], config["latent_dim"] + config["action_dim"]))
    h = mx.zeros((config["B"], config["deter_dim"]))
    
    h_next = cell(x, h)
    assert h_next.shape == (config["B"], config["deter_dim"])
    assert float(mx.sum(h_next)) != 0.0  # State must update away from baseline zeros


def test_conditional_normalizing_flow(config):
    """Test log-probability evaluations and sampling profiles inside RealNVP layers."""
    flow = ConditionalNormalizingFlow(latent_dim=config["latent_dim"], context_dim=config["deter_dim"])
    context = mx.random.normal((config["B"], config["deter_dim"]))
    
    # Assert sample extraction dimensions
    z_sample = flow.sample(context)
    assert z_sample.shape == (config["B"], config["latent_dim"])
    
    # Assert log probability trace profiles match batch layouts
    log_prob = flow.log_prob(z_sample, context)
    assert log_prob.shape == (config["B"],)

# =====================================================================
# 3. TEST WORLD MODEL PIPELINES
# =====================================================================

def test_world_model_loss_compilation(config):
    """Assert World Model computes a single scalar value and broadcasts shapes correctly."""
    wm = ContinuousDreamerWorldModel(
        obs_dim=config["obs_dim"], action_dim=config["action_dim"], 
        deter_dim=config["deter_dim"], latent_dim=config["latent_dim"]
    )
    
    dummy_obs = mx.random.normal((config["B"], config["T"], config["obs_dim"]))
    dummy_acts = mx.random.normal((config["B"], config["T"], config["action_dim"]))
    dummy_rews = mx.random.normal((config["B"], config["T"], 1))
    dummy_conts = mx.ones((config["B"], config["T"], 1))

    # Test forward loss computation returns a valid scalar
    loss = wm.calculate_loss(dummy_obs, dummy_acts, dummy_rews, dummy_conts)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    # Verify compatibility with nn.value_and_grad graph tracing
    loss_fn = nn.value_and_grad(wm, lambda m, o, a, r, c: m.calculate_loss(o, a, r, c))
    val, grads = loss_fn(wm, dummy_obs, dummy_acts, dummy_rews, dummy_conts)
    assert "encoder" in grads
    assert "decoder" in grads

# =====================================================================
# 4. TEST ACTOR-CRITIC HORIZON MANAGEMENT
# =====================================================================

def test_actor_tanh_normal_bounds(config):
    """Verify the continuous policy applies standard Tanh bounding adjustments to actions."""
    actor = ContinuousActor(feature_dim=config["feat_dim"], action_dim=config["action_dim"])
    features = mx.random.normal((config["B"], config["feat_dim"]))
    
    action, log_prob, entropy = actor.sample_and_log_prob(features)
    
    assert action.shape == (config["B"], config["action_space"] if "action_space" in config else config["action_dim"])
    assert log_prob.shape == (config["B"],)
    assert entropy.shape == (config["B"],)
    
    # Tanh actions must be safely bound within [-1.0, 1.0] limits
    assert np.all(np.array(action) >= -1.0)
    assert np.all(np.array(action) <= 1.0)


def test_actor_critic_container_and_loss(config):
    """Assert the container setup correctly passes structural module parameter maps into updates."""
    wm = ContinuousDreamerWorldModel(
        obs_dim=config["obs_dim"], action_dim=config["action_dim"], 
        deter_dim=config["deter_dim"], latent_dim=config["latent_dim"]
    )
    actor = ContinuousActor(feature_dim=config["feat_dim"], action_dim=config["action_dim"])
    critic = ContinuousCritic(feature_dim=config["feat_dim"])
    updater = ActorCriticUpdater(actor, critic, wm)
    
    # Initialize container module matching main architecture fixes
    class ActorCriticContainer(nn.Module):
        def __init__(self, act, crit):
            super().__init__()
            self.actor = act
            self.critic = crit

    ac_container = ActorCriticContainer(actor, critic)
    start_features = mx.random.normal((config["B"] * config["T"], config["feat_dim"]))
    horizon = 12

    # 1. Assert unrolled imagination layout shapes
    imag_features = updater.rollout_imagination(start_features, horizon=horizon)
    assert imag_features.shape == (config["B"] * config["T"], horizon, config["feat_dim"])

    # 2. Check joint loss calculations trace cleanly via nn.value_and_grad
    def ac_loss_fn(m, sf):
        updater.actor = m.actor
        updater.critic = m.critic
        return updater.calculate_losses(sf)

    ac_grad_fn = nn.value_and_grad(ac_container, ac_loss_fn)
    (a_loss, c_loss), ac_grads = ac_grad_fn(ac_container, start_features)
    
    assert a_loss.ndim == 0
    assert c_loss.ndim == 0
    assert "actor" in ac_grads
    assert "critic" in ac_grads

# =====================================================================
# 5. TEST DATA SEQUENCE STORAGE
# =====================================================================

def test_sequence_replay_buffer_slicing(config):
    """Ensure the buffer stores array contexts and samples temporal sub-sequence slices."""
    buffer = SequenceReplayBuffer(capacity=10, seq_len=config["T"])
    
    ep_len = 25
    obs_ep = [np.random.normal(size=(config["obs_dim"],)).astype(np.float32) for _ in range(ep_len)]
    act_ep = [np.random.normal(size=(config["action_dim"],)).astype(np.float32) for _ in range(ep_len)]
    rew_ep = [np.array([1.0], dtype=np.float32) for _ in range(ep_len)]
    cont_ep = [np.array([1.0], dtype=np.float32) for _ in range(ep_len)]
    
    buffer.add_episode(obs_ep, act_ep, rew_ep, cont_ep)
    assert len(buffer) == 1

    # Extract sequences using batch mechanics
    b_obs, b_act, b_rew, b_cont = buffer.sample_batch(batch_size=2)
    
    assert b_obs.shape == (2, config["T"], config["obs_dim"])
    assert b_act.shape == (2, config["T"], config["action_dim"])
    assert b_rew.shape == (2, config["T"], 1)
    assert b_cont.shape == (2, config["T"], 1)
