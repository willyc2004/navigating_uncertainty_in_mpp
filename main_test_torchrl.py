## Imports
import os
import tqdm
import time
from datetime import datetime
import copy

# Datatypes
import yaml
from collections import defaultdict
from typing import Optional, Tuple, Dict
from dotmap import DotMap
from tensordict.nn import TensorDictModule

# Machine learning
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import wandb

# TorchRL
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs
from torchrl.modules import ProbabilisticActor, IndependentNormal, TruncatedNormal, ValueOperator, TanhNormal
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.ddpg import DDPGLoss
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
from environment.utils import compute_violation
from models.encoder import MLPEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache
from models.critic import CriticNetwork
from models.loss import FeasibilityClipPPOLoss

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

def compute_surrogate_loss(ll, td, clip_epsilon, normalize_advantage=False) -> Dict:
    """Compute the surrogate loss for PPO."""
    # Unpack the tensors
    old_ll = td["sample_log_prob"].clone()  # already detached
    advantage = td["advantage"].clone().squeeze(-1)  # already detached
    # Normalize the advantage
    if normalize_advantage:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

    if torch.isnan(ll).any():
        raise ValueError("Log likelihood is NaN.")
    if torch.isnan(old_ll).any():
        raise ValueError("Old log likelihood is NaN.")
    if torch.isnan(advantage).any():
        raise ValueError("Advantage is NaN.")

    # Compute the surrogate loss
    ratio = torch.exp(ll - old_ll)
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    clip_fraction = (torch.abs(ratio - 1) > clip_epsilon).float().mean()
    surrogate_loss = -torch.min(ratio * advantage, clipped_ratio * advantage).mean()

    # if nan, raise error
    if torch.isnan(ratio).any():
        raise ValueError("Ratio is NaN.")
    if torch.isnan(surrogate_loss):
        raise ValueError("Surrogate loss is NaN.")

    # Return dictionary with loss and metrics
    return {"surrogate_loss": surrogate_loss,
            "clip_fraction": clip_fraction,
            "kl_approx": (old_ll - ll).mean(),
            "ratio": ratio.mean(),
            "advantage": advantage.mean(),
            }


def compute_loss_feasibility(td, action, feasibility_coef, aggregate_feasibility="sum"):
    """Compute feasibility loss based on the action, lhs_A, and rhs tensors."""
    lhs_A, rhs = td.get("lhs_A"), td.get("rhs")
    violation = compute_violation(action, lhs_A, rhs)

    # Get aggregation dimensions
    if aggregate_feasibility == "sum":
        sum_dims = [-x for x in range(1, violation.dim())]
        return feasibility_coef * violation.sum(dim=sum_dims).mean(), violation
    elif aggregate_feasibility == "mean":
        return feasibility_coef * violation.mean(), violation

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

## Main function
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
        "temperature": config.model.temperature,
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
    if config.algorithm.type == "sac":
        # Define two Q-networks for the critics
        critic1 = ValueOperator(
            CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=decoder_layers,
                          context_embedding=context_embed, normalization=config.model.normalization,
                          dropout_rate=dropout_rate, temperature=config.model.critic_temperature, customized=True,
                          use_q_value=True, action_dim=action_dim).to(device),
            in_keys=["observation", "action"],  # Input tensor key in TensorDict
            out_keys=["state_action_value"],
        )
        critic2 = copy.deepcopy(critic1)  # Second critic network
        critic = [critic1, critic2]
    else:
        # Get critic
        critic = TensorDictModule(
            CriticNetwork(encoder, embed_dim=embed_dim, hidden_dim=hidden_dim, num_layers=decoder_layers,
                          context_embedding=context_embed, normalization=config.model.normalization,
                          dropout_rate=dropout_rate, temperature=config.model.critic_temperature,
                          customized=True).to(device),
            in_keys=["observation",],  # Input tensor key in TensorDict
            out_keys=["state_value"],  # ["state_action_value"]  # Output tensor key in TensorDict
        )

    # Get ProbabilisticActor (for stochastic policies)
    actor = TensorDictModule(
        Actor(encoder, decoder).to(device),
        in_keys=["observation",],  # Input tensor key in TensorDict
        out_keys=["loc","scale"]  # Output tensor key in TensorDict
    )
    policy = ProbabilisticActor(
        module=actor,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={"low": 0.0, "high": 50.0},
        # distribution_kwargs={"low": 0.0,, "high": 50.0},
        # distribution_kwargs={"scale": 1.0},
        return_log_prob=True,
    )

    ## Main loop
    # Train the model
    if config.model.phase == "train":
        train(policy, critic, **config)
    # Test the model
    elif config.model.phase == "test":
        raise NotImplementedError("Testing not implemented yet.")
    #     datestamp = "20241214_065732"
    #     checkpoint_path = f"./saved_models"
    #     pth_name = f"/trained_model_{datestamp}.pth"
    #     pth = torch.load(checkpoint_path + pth_name,)
    #     model.load_state_dict(pth, strict=True)
    #     # todo: extract trained hyper-parameters
    #     # todo: check if two models can be re-loaded
    #
    #     # Wrap in a TensorDictModule
    #     test_td_module = TensorDictModule(
    #         model,
    #         in_keys=["observation", ],  # Input tensor key in TensorDict
    #         out_keys=["loc"]  # Output tensor key in TensorDict
    #     )
    #
    #     # ProbabilisticActor (for stochastic policies)
    #     test_policy = ProbabilisticActor(
    #         module=test_td_module,
    #         in_keys=["loc", ],
    #         distribution_class=IndependentNormal,
    #         distribution_kwargs={"scale": 1.0}
    #     )
    #
    #
    #     # Initialize policy
    #     env_kwargs["float_type"] = torch.float32
    #     test_env = make_env(env_kwargs)  # Re-initialize the environment
    #
    #     # Run multiple iterations to measure inference time
    #     num_runs = 5
    #     outs = []
    #     returns = []
    #     times = []
    #     revenues = []
    #     costs = []
    #     violations = []
    #
    #     init_td = test_env.generator(batch_size).clone()
    #     for i in range(num_runs):
    #         # Set a new seed for each run
    #         print(f"Run {i + 1}/{num_runs}")
    #         seed = i
    #         torch.manual_seed(seed)
    #
    #         # Sync GPU before starting timer if using CUDA
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         start_time = time.perf_counter()
    #
    #         # Run inference
    #         td = test_env.reset(test_env.generator(batch_size=batch_size, td=init_td), )
    #         rollout = test_env.rollout(env.K*env.T, test_policy, tensordict=td, auto_reset=True)
    #         # todo: add rollout_results to outs
    #
    #         # Sync GPU again after inference if using CUDA
    #         if torch.cuda.is_available():
    #             torch.cuda.synchronize()
    #         end_time = time.perf_counter()
    #
    #         # Calculate and record inference time for each run
    #         returns.append(rollout["next", "reward"].mean())
    #         times.append(end_time - start_time)
    #         revenues.append(rollout[..., -1]["next", "state", "total_revenue"].mean())
    #         costs.append(rollout[..., -1]["next", "state", "total_cost"].mean())
    #         violations.append(rollout[..., -1]["next", "state", "total_violation"].sum(dim=(-1, -2)).mean())
    #     returns = torch.tensor(returns)
    #     times = torch.tensor(times)
    #     revenues = torch.tensor(revenues)
    #     costs = torch.tensor(costs)
    #     violations = torch.tensor(violations)
    #     print("-"*50)
    #     print(f"Mean return: {returns.mean():.4f}")
    #     print(f"Mean inference time: {times.mean():.4f} seconds")
    #     print(f"Mean revenue: {revenues.mean():.4f}")
    #     print(f"Mean cost: {costs.mean():.4f}")
    #     print(f"Mean violation: {violations.mean():.4f}")
    #
    #     # rollout_results(test_env, outs, td, batch_size, checkpoint_path,
    #     #                 am_ppo_params["projection_type"], config["env"]["utilization_rate_initial_demand"], times)

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
    val_data_size = kwargs["training"]["val_data_size"]
    validation_freq = kwargs["training"]["validation_freq"]
    tau = 0.005 # soft-updates

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
    # advantage_module = GAE(
    #     gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True
    # )
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
    elif kwargs["algorithm"]["type"] == "sac":
        # Create the SAC loss module
        # loss_module = SACLoss(
        #     actor_network=policy,
        #     qvalue_network=critic,  # List of two Q-networks
        # )
        critic1 = critic[0]
        critic2 = critic[1]
        target_critic1 = copy.deepcopy(critic1).to(device)
        target_critic2 = copy.deepcopy(critic2).to(device)

    elif kwargs["algorithm"]["type"] == "ddpg":
        # Create the DDPG loss module
        loss_module = DDPGLoss(
            actor_network=policy,
            value_network=critic,
            delay_actor=True,
            delay_value=True,
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
            feasibility_coef=feasibility_lambda,
            normalize_advantage=True,
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
    actor_optim = torch.optim.Adam(policy.parameters(), lr=lr)
    if kwargs["algorithm"]["type"] == "sac":
        critic_optim = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=lr)
    else:
        critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
    train_updates = train_data_size // (batch_size * n_step)
    pbar = tqdm.tqdm(range(train_updates))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, train_data_size)

    # Training loop
    # torch.autograd.set_detect_anomaly(True)
    for step, td in enumerate(collector):
        # for _ in range(num_epochs):
            # advantage_module(td)
        replay_buffer.extend(td)
        for _ in range(batch_size // mini_batch_size):
            # Sample mini-batch (including actions, n-step returns, old log likelihoods, target_values)
            subdata = replay_buffer.sample(mini_batch_size).to(device)

            ## Loss optimization
            loss_out = {}
            # Critic loss calculation
            with torch.no_grad():
                # Nex action
                # print(policy)
                next_policy_out = policy(subdata)
                next_action = next_policy_out["action"]
                target_q1 = target_critic1(next_policy_out["observation"], next_action)
                target_q2 = target_critic2(next_policy_out["observation"], next_action)
                target_q_min = torch.min(target_q1, target_q2) - entropy_lambda * next_policy_out["sample_log_prob"].unsqueeze(-1)
                target_value = subdata["next", "reward"] + (1 - subdata["done"].float()) * gamma * target_q_min

            # Current action
            current_q1 = critic1(subdata["observation"], subdata["action"])
            current_q2 = critic2(subdata["observation"], subdata["action"])

            # Update critic
            loss_out["loss_critic"] = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
            critic_optim.zero_grad()
            loss_out["loss_critic"].backward()
            torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_grad_norm)
            critic_optim.step()

            # Soft update target critics
            for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # Compute Actor Loss
            policy_out = policy(subdata)
            new_action = policy_out["action"]
            log_prob = policy_out["sample_log_prob"].unsqueeze(-1)

            # Feasibility loss
            loss_out["loss_feasibility"], loss_out["violation"] = \
                compute_loss_feasibility(policy_out, new_action, feasibility_lambda, "sum")
            q1 = critic1(policy_out["observation"], new_action)
            q2 = critic2(policy_out["observation"], new_action)
            q_min = torch.min(q1, q2)

            # Update actor
            loss_out["loss_actor"] = (entropy_lambda * log_prob - q_min).mean() + loss_out["loss_feasibility"]
            actor_optim.zero_grad()
            loss_out["loss_actor"].backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            actor_optim.step()

        # Log metrics
        pbar.update(1)
        log = {
            # General
            "step": step,
            # Losses
            # "loss": loss.item(),
            "loss_actor": loss_out["loss_actor"],
            "loss_critic":  loss_out["loss_critic"],
            "loss_feasibility":loss_out["loss_feasibility"],
            # "loss_entropy": loss_out["loss_entropy"],
            # Return, gradient norm and loss support
            "return": subdata['next', 'reward'].mean().item(),
            # "grad_norm": gn.item(),
            # "clip_fraction": loss_out["clip_fraction"],
            # todo: add kl_approx, ratio, advantage
            # Constraints
            "mean_total_violation": loss_out["violation"].sum(dim=(-2,-1)).mean().item(),
            "total_violation": policy_out['violation'].sum(dim=(-2,-1)).mean().item(),
            "demand_violation": policy_out['violation'][...,0].sum(dim=(1)).mean().item(),
            "capacity_violation": policy_out['violation'][...,1:-4].sum(dim=(1)).mean().item(),
            "LCG_violation": policy_out['violation'][..., -4:-2].sum(dim=(1,2)).mean().item(),
            "VCG_violation": policy_out['violation'][..., -2:].sum(dim=(1,2)).mean().item(),

            # Environment
            "total_revenue": subdata["state", "total_revenue"][...,-1].mean().item(),
            "total_cost": subdata["state", "total_cost"][...,-1].mean().item(),
            "total_loaded": subdata["state", "total_loaded"][...,-1].mean().item(),
            "total_demand":subdata['realized_demand'][:,0,:].sum(dim=-1).mean(),
            "total_e[x]_demand": td['init_expected_demand'][:, 0, :].sum(dim=-1).mean(),
        }
        # Log metrics
        pbar.set_description(
            # Loss, gn and rewards
            f"return: {log['return']: 4.4f}, "
            # f"loss:  {log['loss']: 4.4f}, "
            f"loss_actor:  {log['loss_actor']: 4.4f}, "
            f"loss_critic:  {log['loss_critic']: 4.4f}, "
            f"mean_violation: {log['mean_total_violation']: 4.4f}, "                
            f"feasibility_loss: {log['loss_feasibility']: 4.4f}, "
            # f"gradient norm: {log['grad_norm']: 4.4}, "
            # Performance
            f"total_revenue: {log['total_revenue']: 4.4f}, "
            f"total_cost: {log['total_cost']: 4.4f}, "
            f"violation: {log['total_violation']: 4.4f}, "
        )
        wandb.log(log)
        # scheduler.step()

        # # Validation
        # if (step + 1) % (train_updates * validation_freq) == 0:
        #     # todo: add validation here for every
        #     validate_policy(env, policy, num_episodes=val_data_size//batch_size)

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model checkpoint with timestamp
    policy_save_path = f"saved_models/trained_policy_{timestamp}.pth"
    critic_save_path = f"saved_models/trained_critic_{timestamp}.pth"
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)
    torch.save(policy.state_dict(), policy_save_path)
    torch.save(critic.state_dict(), critic_save_path)

    # Log the model checkpoint to wandb
    wandb.save(policy_save_path)
    wandb.save(critic_save_path)

    # Close environments
    env.close()

## Validation
def validate_policy(env: EnvBase, policy_module: ProbabilisticActor, num_episodes: int = 10, device: str = "cuda"):
    """
    Perform validation rollouts for a given policy in an environment.

    Args:
        env (EnvBase): The environment for validation.
        policy_module (ProbabilisticActor): The trained policy module.
        num_episodes (int): Number of episodes to run for validation.
        device (str): Device to run the policy ('cpu' or 'cuda').

    Returns:
        float: The average reward over the validation episodes.
    """
    policy_module.eval()  # Set the policy to evaluation mode
    total_rewards = []

    with torch.no_grad():
        for episode in range(num_episodes):
            # Reset the environment and get the initial observation
            tensordict = env.reset()
            tensordict = tensordict.to(device)
            done = False
            episode_reward = 0

            while not done:
                # Get action distribution from the policy and sample an action
                action_dist = policy_module(tensordict)
                action = action_dist.sample()
                tensordict.set("action", action)

                # Step the environment
                next_tensordict = env.step(tensordict)

                # Accumulate the reward
                reward = next_tensordict["reward"]
                episode_reward += reward.item()

                # Update the current state for the next step
                tensordict = next_tensordict

                # Check if the episode is done
                done = next_tensordict["done"].item()

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1}/{num_episodes}: Reward = {episode_reward}")

    # Compute the average reward
    avg_reward = sum(total_rewards) / num_episodes
    print(f"\nAverage Reward over {num_episodes} Episodes: {avg_reward:.2f}")

    policy_module.train()  # Set the policy back to training mode
    return avg_reward

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