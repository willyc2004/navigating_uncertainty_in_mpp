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
from models.critic import CriticNetwork
from models.ppo_feas_loss import FeasibilityClipPPOLoss

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
class Actor(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden, init_embed = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out

## Training
def train(policy, critic, device=torch.device("cuda"), **kwargs):
    # todo: extend with mini-batch training, data loader, REINFORCE, PPO, etc.
    # Hyperparameters
    batch_size = kwargs["model"]["batch_size"]
    mini_batch_size = int(kwargs["algorithm"]["mini_batch_size"] * batch_size)
    n_step = kwargs["algorithm"]["n_step"]
    num_epochs = kwargs["algorithm"]["ppo_epochs"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    vf_lambda = kwargs["algorithm"]["vf_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]
    lr = kwargs["training"]["lr"]
    train_data_size = kwargs["training"]["train_data_size"]

    # Environment
    env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)

    # Optimizer, loss module, data collector, and scheduler
    # if kwargs["algorithm"]["name"] == "reinforce":
    #     loss_module = ReinforceLoss(actor_network=policy, critic_network=critic,)
    # elif kwargs["algorithm"]["name"] == "ppo":
    gamma = kwargs["algorithm"]["gamma"]
    gae_lambda = kwargs["algorithm"]["gae_lambda"]
    clip_epsilon = kwargs["algorithm"]["clip_range"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]

    # Loss modules
    advantage_module = GAE(
        gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True
    )
    if kwargs["algorithm"]["type"] == "reinforce":
        loss_module = ReinforceLoss(
            actor_network=policy,
            critic_network=critic,
        )
    elif kwargs["algorithm"]["type"] == "ppo":
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_lambda),
            entropy_coef=entropy_lambda,
            critic_coef=vf_lambda,
            loss_critic_type="smooth_l1",
        )
    elif kwargs["algorithm"]["type"] == "ppo_feas":
        loss_module = FeasibilityClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_lambda),
            entropy_coef=entropy_lambda,
            critic_coef=vf_lambda,
            loss_critic_type="smooth_l1",
            feasibility_lambda=feasibility_lambda,
        )
    else:
        raise ValueError(f"Algorithm {kwargs['algorithm']['type']} not recognized.")

    # Data collector and replay buffer
    collector = SyncDataCollector(
        env,
        policy,
        frames_per_batch=batch_size*n_step, # batch_size * steps_per_episode
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

    pbar = tqdm.tqdm(range(len(collector)))
    for step, td in enumerate(collector):
        for _ in range(num_epochs):
            advantage_module(td)
            replay_buffer.extend(td)
            for _ in range(batch_size // mini_batch_size):
                subdata = replay_buffer.sample(mini_batch_size)
                # print(subdata.keys())
                for k, v in subdata.items():
                    if v.requires_grad:
                        print(k, v.requires_grad)

                loss_vals = loss_module(subdata.to(device))
                print(loss_vals.keys())
                for k, v in loss_vals.items():
                    if v.requires_grad:
                        print(k, v.requires_grad)

                # print(loss_vals["loss_objective"].requires_grad)
                breakpoint()
                loss_vals["violation"] = subdata["next", "violation"]
                loss_vals["loss_feasibility"] = feasibility_lambda * loss_vals["violation"].sum(dim=(-2,-1)).mean()
                loss = (loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                        + loss_vals["loss_feasibility"]
                )

                # Optimization: backward, grad clipping and optimization step
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

            # Log metrics
            pbar.update(1)
            log = {
                # General
                "step": step,
                # Losses
                "loss": loss.item(),
                "loss_actor": loss_vals["loss_objective"],
                "loss_critic": loss_vals["loss_critic"],
                "loss_feasibility":loss_vals["loss_feasibility"],
                "loss_entropy": loss_vals["loss_entropy"],
                # Gradient, return and loss support
                "grad_norm": gn.item(),
                "return":subdata['next', 'reward'].sum(dim=(-2,-1)).mean().item(),
                "kl_approx": loss_vals["loss_entropy"],
                "clip_fraction": loss_vals["clip_fraction"],
                "entropy": loss_vals["entropy"],
                # Constraints
                "total_violation": loss_vals["violation"].sum(dim=(-2,-1)).mean().item(),
                "demand_violation":loss_vals["violation"][...,0].sum(dim=(1)).mean().item(),
                "LCG_violation": loss_vals["violation"][..., 1:3].sum(dim=(1,2)).mean().item(),
                "VCG_violation": loss_vals["violation"][..., 3:5].sum(dim=(1,2)).mean().item(),

                # Environment
                "total_revenue": subdata["next", "state", "total_revenue"][...,-1].mean().item(),
                "total_cost": subdata["next", "state", "total_cost"][...,-1].mean().item(),
                "total_loaded": subdata["next", "state", "total_loaded"][...,-1].mean().item(),
                "total_demand":subdata['realized_demand'][:,0,:].sum(dim=-1).mean(),
                "total_e[x]_demand": subdata['init_expected_demand'][:, 0, :].sum(dim=-1).mean(),
            }
            # Log metrics
            pbar.set_description(
                # Loss, gn and rewards
                f"return: {log['return']: 4.4f}, "
                f"loss:  {log['loss']: 4.4f}, "
                f"violation: {log['total_violation']: 4.4f}, "                
                f"feasibility_loss: {log['loss_feasibility']: 4.4f}, "
                f"gradient norm: {log['grad_norm']: 4.4}, "
            )
        wandb.log(log)
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
    # todo: fix e[x] demand being static?
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
    critic = CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=1,
                           context_embedding=context_embed, normalization=config.model.normalization,
                           dropout_rate=dropout_rate, customized=True).to(device)
    actor = Actor(encoder, decoder).to(device)

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
        "batch_size": config.model.batch_size,
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
        # todo: extract trained hyper-parameters
        # todo: check if two models can be re-loaded

        # Wrap in a TensorDictModule
        test_td_module = TensorDictModule(
            model,
            in_keys=["observation", ],  # Input tensor key in TensorDict
            out_keys=["loc"]  # Output tensor key in TensorDict
        )

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