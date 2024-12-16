## Imports
import os
import tqdm
import time
from datetime import datetime

# Datatypes
import yaml
from collections import defaultdict
from typing import Optional
from dotmap import DotMap
from tensordict.nn import TensorDictModule

# Machine learning
import random
import numpy as np
import torch
from torch import nn
import wandb

# TorchRL
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, IndependentNormal
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# RL4CO
from rl4co.models.zoo.am.encoder import AttentionModelEncoder

# Custom
from environment.env_torchrl import MasterPlanningEnv
from environment.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from models.encoder import MLPEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache

## Helper functions
def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs:DotMap, batch_size:Optional[list] = [], device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    return MasterPlanningEnv(batch_size=batch_size, **env_kwargs).to(device)  # Custom environment

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # He initialization for ReLU
    if isinstance(m, torch.nn.MultiheadAttention):
        torch.nn.init.normal_(m.in_proj_weight, mean=0.0, std=0.01)  # Small normal init for attention weights
    if isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1.0)
        torch.nn.init.constant_(m.bias, 0.0)

## Classes
class SimplePolicy(nn.Module):
    def __init__(self, hidden_dim, act_dim, device: torch.device = torch.device("cuda"), dtype=torch.float32):
        super().__init__()
        self.net = nn.Sequential(
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(hidden_dim),
        nn.ReLU(),
        nn.LazyLinear(act_dim),
    ).to(device).to(dtype)

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    """Create encoder model from initial embedding, encoder models passed as arguments."""
    def __init__(self, init_embed, encoder):
        super().__init__()
        self.init_embed = init_embed
        self.encoder = encoder

    def forward(self, obs):
        # Initial embedding
        init_embed = self.init_embed(obs)
        # Encode initial embedding to hidden state
        hidden, _ = self.encoder(init_embed)
        return hidden

class ActorDecoder(nn.Module):
    """Create decoder model from context embedding, decoder models passed as arguments."""
    def __init__(self, context_embed, decoder):
        super().__init__()
        self.context_embed = context_embed
        self.decoder = decoder

    def forward(self, obs, hidden):
        # Context embedding
        context_embed = self.context_embed(obs, hidden)
        # Decode context embedding to output
        dec_out = self.decoder(context_embed)
        return dec_out

class CriticDecoder(nn.Module):
    """Create decoder model from context embedding, decoder models passed as arguments."""
    def __init__(self, context_embed, decoder):
        super().__init__()
        self.context_embed = context_embed
        self.decoder = decoder

    def forward(self, obs, hidden):
        # Context embedding
        context_embed = self.context_embed(obs, hidden)
        # Decode context embedding to output
        dec_out = self.decoder(context_embed)
        return dec_out

class Actor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out

class Critic(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out

## Training
def train(policy, critic, device=torch.device("cuda"), **kwargs):
    # todo: extend with mini-batch training, data loader, REINFORCE, PPO, etc.
    # Hyperparameters
    lr = kwargs["training"]["lr"]
    batch_size = kwargs["model"]["batch_size"]
    mini_batch_size = int(kwargs["training"]["mini_batch_size"] * batch_size)
    train_data_size = kwargs["training"]["train_data_size"]
    num_epochs = kwargs["algorithm"]["ppo_epochs"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    vf_lambda = kwargs["algorithm"]["vf_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]

    # Environment
    env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)

    # Optimizer, loss module, data collector, and scheduler
    # if kwargs["algorithm"]["name"] == "reinforce":
    #     loss_module = ReinforceLoss(actor_network=policy, critic_network=critic,)
    # elif kwargs["algorithm"]["name"] == "ppo":
    gamma = kwargs["algorithm"]["gamma"]
    gae_lambda = kwargs["algorithm"]["gae_lambda"]
    clip_epsilon = kwargs["algorithm"]["clip_epsilon"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]


    advantage_module = GAE(
        gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_lambda),
        entropy_coef=entropy_lambda,
        # these keys match by default but we set this for completeness
        critic_coef=vf_lambda,
        loss_critic_type="smooth_l1",
    )

    # Data collector and replay buffer
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=batch_size,
        total_frames=train_data_size,
        split_trajs=False,
        device=device,
    )
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=batch_size),
        sampler=SamplerWithoutReplacement(),
    )

    # Optimizer and scheduler
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, train_data_size)

    pbar = tqdm.tqdm(range(train_data_size // batch_size))
    for step, td in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(td)
            data_view = td.reshape(-1)
            replay_buffer.extend(data_view)
            for _ in range(batch_size // mini_batch_size):
                subdata = replay_buffer.sample(mini_batch_size)
                loss_vals = loss_module(subdata.to(device))
                violation = subdata["next", "state", "violation"]
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                        + feasibility_lambda * violation.mean()
                )

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

            # Log metrics
            pbar.update(td.numel())

            # # Log metrics
            # pbar.set_description(
            #     # Loss, gn and rewards
            #     f"traj_return: {traj_return.mean(): 4.4f}, "
            #     f"last reward: {rollout[..., -1]['next', 'reward'].mean(): 4.4f}, "
            #     f"loss :  {loss: 4.4f}, "
            #     f"gradient norm: {gn: 4.4}, "
            #     # Constraints
            #     f"total_violation: {rollout[..., -1]['next', 'state', 'total_violation'].sum(dim=-1).mean(): 4.4f}, "
            #     f"demand_violation: {rollout[..., -1]['next', 'state', 'total_violation'][...,0].mean(): 4.4f}, "
            #     f"LCG_violation: {rollout[..., -1]['next', 'state', 'total_violation'][...,1:3].sum(dim=-1).mean(): 4.4f}, "
            #     f"VCG_violation: {rollout[..., -1]['next', 'state', 'total_violation'][...,3:5].sum(dim=-1).mean(): 4.4f}, "
            #     # Env
            #     f"total_revenue: {rollout[..., -1]['next', 'state', 'total_revenue'].mean(): 4.4f}, "
            #     f"total_cost: {rollout[..., -1]['next', 'state', 'total_cost'].mean(): 4.4f}, "
            #     f"total_loaded: {rollout[..., -1]['next', 'state', 'total_loaded'].mean(): 4.4f}, "
            #     f"total demand: {rollout[..., -1]['next', 'realized_demand'].sum(dim=-1).mean(): 4.4f}, "
            #     f"total e[x] demand: {init_td['expected_demand'].sum(dim=-1).mean(): 4.4f}, "
            # )
        # log = {
        #     # General
        #     "step": step,
        #     # Trajectory
        #     "traj_return": traj_return.mean().item(),
        #     "traj_violation": traj_violation.mean().item(),
        #     "last_reward": rollout[..., -1]["next", "reward"].mean().item(),
        #     # Loss and gradients
        #     "loss": loss.item(),
        #     "grad_norm": gn.item(),
        #     # Constraints
        #     "total_violation": rollout[..., -1]["next", "state", "total_violation"].sum(dim=-1).mean().item(),
        #     "demand_violation": rollout[..., -1]["next", "state", "total_violation"][...,0].mean().item(),
        #     "LCG_violation": rollout[..., -1]["next", "state", "total_violation"][..., 1:3].sum(dim=-1).mean().item(),
        #     "VCG_violation": rollout[..., -1]["next", "state", "total_violation"][..., 3:5].sum(dim=-1).mean().item(),
        #
        #     # Environment
        #     "total_revenue": rollout[..., -1]["next", "state", "total_revenue"].mean().item(),
        #     "total_cost": rollout[..., -1]["next", "state", "total_cost"].mean().item(),
        #     "total_loaded": rollout[..., -1]["next", "state", "total_loaded"].mean().item(),
        #     "total_demand":rollout[..., -1]['next', 'realized_demand'].sum(dim=-1).mean().item(),
        #     "total_e_x_demand": init_td['expected_demand'].sum(dim=-1).mean().item(),
        # }
        # wandb.log(log)
        scheduler.step()

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model checkpoint with timestamp
    model_save_path = f"saved_models/trained_model_{timestamp}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)

    # Log the model checkpoint to wandb
    wandb.save(model_save_path)


def main(config: Optional[DotMap] = None):
    # todo: clean-up and refactor all hyperparameters etc.
    # Environment kwargs
    env_kwargs = config.env

    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch._dynamo.config.cache_size_limit = 64  # or some higher value
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## Environment initialization
    env = make_env(env_kwargs)
    seed = env_kwargs.seed
    env.set_seed(seed)
    # env = ParallelEnv(4, make_env)     # todo: fix parallel env
    check_env_specs(env)  # this must pass for ParallelEnv to work

    ## Model initialization
    # Embedding dimensions
    init_dim = config.model.init_dim
    embed_dim = config.model.embed_dim
    sequence_dim = env.K * env.T
    obs_dim = env.observation_spec["observation"].shape[0]
    action_dim = env.action_spec.shape[0]
    # Embedding initialization
    init_embed = MPPInitEmbedding(embed_dim, action_dim, env)
    context_embed = MPPContextEmbedding(action_dim, embed_dim, env, config.model.demand_aggregation)
    dynamic_embed = StaticEmbedding(obs_dim, embed_dim)

    # Model initialization
    hidden_dim = config.model.feedforward_hidden
    encoder_layers = config.model.num_encoder_layers
    decoder_layers = config.model.num_decoder_layers
    num_heads = config.model.num_heads
    dropout_rate = config.model.dropout_rate
    decoder_args = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "num_hidden_layers": decoder_layers,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate,
        "action_size": action_dim,
        "state_size": obs_dim,
        "total_steps": sequence_dim,
        "init_embedding": init_embed,
        "context_embedding": context_embed,
        "dynamic_embedding": dynamic_embed,
        "normalization": config.model.normalization,
    }
    encoder_args = {
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "init_embedding": init_embed,
        "env_name": env.name,
        "num_layers": encoder_layers,
        "feedforward_hidden": hidden_dim,
        "normalization": config.model.normalization,
    }

    # Setup model
    # todo: fix e[x] being static?

    if config.model.encoder_type == "attention":
        encoder = AttentionModelEncoder(**encoder_args,)
    else:
        encoder = MLPEncoder(**encoder_args)
    if config.model.decoder_type == "attention":
        decoder = AttentionDecoderWithCache(**decoder_args)
    elif config.model.decoder_type == "mlp":
        decoder = MLPDecoderWithCache(**decoder_args)
    else:
        raise ValueError(f"Decoder type {config.model.decoder_type} not recognized.")

    encoder = Encoder(init_embed, encoder).to(device)
    actor_decoder = ActorDecoder(context_embed, decoder).to(device)
    critic_decoder = CriticDecoder(context_embed, decoder).to(device)
    actor = Actor(encoder, actor_decoder).to(device)
    critic = Critic(encoder, critic_decoder).to(device)
    # model = SimplePolicy(hidden_dim=hidden_dim, act_dim=action_dim, device=device, dtype=env.float_type).apply(init_weights)

    # Get ProbabilisticActor (for stochastic policies)
    actor = TensorDictModule(
        actor,
        in_keys=["observation",],  # Input tensor key in TensorDict
        out_keys=["loc"]  # Output tensor key in TensorDict
    )
    policy = ProbabilisticActor(
        module=actor,
        in_keys=["loc", ],
        distribution_class=IndependentNormal,
        distribution_kwargs={"scale": 1.0},
        return_log_prob=True,
    )
    # Get critic
    critic = TensorDictModule(
        critic,
        in_keys=["observation", ],  # Input tensor key in TensorDict
        out_keys=["state_value"]  # Output tensor key in TensorDict
    )

    ## Hyperparameters
    batch_size = config.model.batch_size
    train_data_size = config.am_ppo.train_data_size
    lr = config.am_ppo.lr

    # AM Model initialization
    model_params = {
        "decoder": decoder,
        "encoder": encoder,
        "init_embedding": init_embed,
        "context_embedding": context_embed,
        "dynamic_embedding": dynamic_embed,
        "projection_type": config.am_ppo.projection_type,
        "projection_kwargs": config.am_ppo.projection_kwargs,
        "select_obs_td":["obs", "done", "timestep", "action_mask", "lhs_A", "rhs", "clip_min", "clip_max",
                              "reward",
                              ("state", "utilization"), ("state", "target_long_crane"), ("state", "total_loaded"),
                              ("state", "total_revenue"), ("state", "total_cost"), ("state", "total_rc"),
                              "realized_demand",
                              ],
        **config.model
    }
    am_ppo_params = {
        "env": env,
        "policy":policy,
        "critic": critic,
        # "critic_kwargs": {"embed_dim": embed_dim, "hidden_dim": hidden_dim, "customized": False},
        # "projection_kwargs": projection_kwargs,
        "decoder_type": config.model.decoder_type,
        "batch_size": batch_size,
        "env_kwargs": env_kwargs,
        "model_kwargs": model_params,
        **config.ppo,
        **config.am_ppo,
    }

    ## Main loop
    # Train the model
    if config.model.phase == "train":
        train(policy, critic, **config)
    # Test the model
    elif config.model.phase == "test":
        datestamp = "20241214_065732"
        checkpoint_path = f"./saved_models"
        pth_name = f"/trained_model_{datestamp}.pth"
        pth = torch.load(checkpoint_path + pth_name,)
        model.load_state_dict(pth, strict=True)
        # Wrap in a TensorDictModule
        test_td_module = TensorDictModule(
            model,
            in_keys=["observation", ],  # Input tensor key in TensorDict
            out_keys=["loc"]  # Output tensor key in TensorDict
        )
        # todo: check if two models can be re-loaded

        # ProbabilisticActor (for stochastic policies)
        test_policy = ProbabilisticActor(
            module=test_td_module,
            in_keys=["loc", ],
            distribution_class=IndependentNormal,
            distribution_kwargs={"scale": 1.0}
        )

        # Initialize policy
        env_kwargs["float_type"] = torch.float32
        test_env = make_env(env_kwargs)  # Re-initialize the environment

        # Run multiple iterations to measure inference time
        num_runs = 5
        outs = []
        returns = []
        times = []
        revenues = []
        costs = []
        violations = []

        init_td = test_env.generator(batch_size).clone()
        for i in range(num_runs):
            # Set a new seed for each run
            print(f"Run {i + 1}/{num_runs}")
            seed = i
            torch.manual_seed(seed)

            # Sync GPU before starting timer if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Run inference
            td = test_env.reset(test_env.generator(batch_size=batch_size, td=init_td), )
            rollout = test_env.rollout(env.K*env.T, test_policy, tensordict=td, auto_reset=True)
            # todo: add rollout_results to outs

            # Sync GPU again after inference if using CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Calculate and record inference time for each run
            returns.append(rollout["next", "reward"].mean())
            times.append(end_time - start_time)
            revenues.append(rollout[..., -1]["next", "state", "total_revenue"].mean())
            costs.append(rollout[..., -1]["next", "state", "total_cost"].mean())
            violations.append(rollout[..., -1]["next", "state", "total_violation"].sum(dim=(-1, -2)).mean())
        returns = torch.tensor(returns)
        times = torch.tensor(times)
        revenues = torch.tensor(revenues)
        costs = torch.tensor(costs)
        violations = torch.tensor(violations)
        print("-"*50)
        print(f"Mean return: {returns.mean():.4f}")
        print(f"Mean inference time: {times.mean():.4f} seconds")
        print(f"Mean revenue: {revenues.mean():.4f}")
        print(f"Mean cost: {costs.mean():.4f}")
        print(f"Mean violation: {violations.mean():.4f}")

        # rollout_results(test_env, outs, td, batch_size, checkpoint_path,
        #                 am_ppo_params["projection_type"], config["env"]["utilization_rate_initial_demand"], times)

if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config_torchrl.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    # Call your main() function
    try:
        wandb.init(config=config,)
        model = main(config)
    except Exception as e:
        # Log the error to WandB
        wandb.log({"error": str(e)})

        # Optionally, use WandB alert for critical errors
        wandb.alert(
            title="Training Error",
            text=f"An error occurred during training: {e}",
            level="error"  # 'info' or 'warning' levels can be used as needed
        )

        # Print the error for local console logging as well
        print(f"An error occurred during training: {e}")
    finally:
        wandb.finish()