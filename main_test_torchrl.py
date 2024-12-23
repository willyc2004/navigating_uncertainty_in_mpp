## Imports
import os
import tqdm
import time
from datetime import datetime
import copy

# Datatypes
import yaml
from collections import defaultdict
from dotmap import DotMap
from typing import Optional, Tuple, Dict, Union, Sequence
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

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
from torchrl.modules import ProbabilisticActor, IndependentNormal, TruncatedNormal, ValueOperator, TanhNormal, MLP
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.tensor_specs import Composite, TensorSpec

# RL4CO
from rl4co.models.zoo.am.encoder import AttentionModelEncoder

# Custom
from environment.env_torchrl import MasterPlanningEnv
from environment.env_port_torchrl import PortMasterPlanningEnv
from environment.embeddings import MPPInitEmbedding, StaticEmbedding, MPPContextEmbedding
from environment.utils import compute_violation
from models.encoder import MLPEncoder
from models.decoder import AttentionDecoderWithCache, MLPDecoderWithCache
from models.critic import CriticNetwork
from models.loss import FeasibilityClipPPOLoss
from models.projection import ProjectionFactory

## Helper functions
def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs:DotMap, batch_size:Optional[list] = [], device: torch.device = torch.device("cuda")):
    """Setup and transform the Pendulum environment."""
    return MasterPlanningEnv(batch_size=batch_size, **env_kwargs).to(device)

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

# todo: redudant - also in loss_module of PPO_feas
def compute_loss_feasibility(td, action, feasibility_coef, aggregate_feasibility="sum"):
    """Compute feasibility loss based on the action, lhs_A, and rhs tensors."""
    lhs_A = td.get("lhs_A")
    rhs = td.get("rhs")
    violation = compute_violation(action, lhs_A, rhs)

    # Get aggregation dimensions
    if aggregate_feasibility == "sum":
        sum_dims = [-x for x in range(1, violation.dim())]
        return feasibility_coef * violation.sum(dim=sum_dims).mean(), violation
    elif aggregate_feasibility == "mean":
        return feasibility_coef * violation.mean(), violation

## Classes
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden, init_embed = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out


class ActorMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ActorMLP, self).__init__()
        # Ensure hidden_dims is a list
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]

        # Create a list to hold all layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))

        # Combine all layers into a Sequential module
        self.hidden_layers = nn.Sequential(*layers)

        # Output layers for mean and standard deviation
        self.mean = nn.Linear(hidden_dims[-1], output_dim)
        self.std = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Pass input through hidden layers
        x = self.hidden_layers(x)
        x = F.relu(x)  # Apply ReLU to the output of the last hidden layer

        # Compute mean and standard deviation
        mean = self.mean(x)
        std = torch.exp(self.std(x))  # Ensure std is positive

        return mean, std

class ProjectionProbabilisticActor(ProbabilisticActor):
    def __init__(self,
                 module: TensorDictModule,
                 in_keys: Union[NestedKey, Sequence[NestedKey]],
                 out_keys: Optional[Sequence[NestedKey]] = None,
                 *,
                 spec: Optional[TensorSpec] = None,
                 projection_layer: Optional[nn.Module] = None,
                 action_rescale_min: Optional[float] = None,
                 action_rescale_max: Optional[float] = None,
                 **kwargs):
        super().__init__(module, in_keys, out_keys, spec=spec, **kwargs)

        # Rescale actions from [-upscale, upscale] to (min, max)
        self.upscale = kwargs.get("distribution_kwargs", {}).get("upscale", 1.0)
        if action_rescale_min is not None and action_rescale_max is not None:
            self.rescale_action = lambda x: x * (action_rescale_max - action_rescale_min) + action_rescale_min
        # Initialize projection layer
        self.projection_layer = projection_layer

    @staticmethod
    def conditional_softmax(sample, upper_bound):
        if sample.dim() - upper_bound.dim() == 1:
            upper_bound = upper_bound.unsqueeze(-1)
        elif sample.dim() - upper_bound.dim() < 0 or sample.dim() - upper_bound.dim() > 1:
            raise ValueError(f"Sample dim {sample.dim()} and upper_bound dim {upper_bound.dim()} not compatible.")

        # If the sum exceeds the upper_bound, apply softmax scaling
        condition = sample.sum(dim=-1, keepdim=True) > upper_bound
        scaled_sample = torch.where(
            condition,
            F.softmax(sample, dim=-1) * upper_bound,
            sample
        )
        return scaled_sample

    def forward(self, *args, **kwargs):
        out = super().forward(*args, **kwargs)
        ub = out["realized_demand"][...,out["timestep"][0]] if out["realized_demand"].dim() == 2 else out["realized_demand"][..., out["timestep"][0,0],:]
        out["action"] = self.conditional_softmax(out["action"], upper_bound=ub).clone()
        if not self.training:
            out["action"] = out["loc"]
        #
        #     if self.rescale_action is not None:
        #         out["action"] = self.rescale_action((out["loc"] + self.upscale) / 2)
        # else:
        #     if self.rescale_action is not None:
        #         out["action"] = self.rescale_action((out["action"] + self.upscale) / 2)  # Rescale from [-x, x] to (min, max)
        if self.projection_layer is not None:
            action = self.projection_layer(out["action"], out["lhs_A"], out["rhs"])
            out["action"] = action
        return out

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
    embed_dim = config.model.embed_dim
    obs_dim = env.observation_spec["observation"].shape[0]
    action_dim = env.action_spec.shape[0]
    sequence_dim = env.K * env.T if env.action_spec.shape[0] == env.B*env.D else env.P-1
    # Embedding initialization
    init_embed = MPPInitEmbedding(obs_dim, action_dim, embed_dim, sequence_dim, env)
    context_embed = MPPContextEmbedding(obs_dim, action_dim, embed_dim, sequence_dim, env, config.model.demand_aggregation)
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
        "scale_max":config.model.scale_max,
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
        Autoencoder(encoder, decoder).to(device),
        in_keys=["observation",],  # Input tensor key in TensorDict
        out_keys=["loc","scale"]  # Output tensor key in TensorDict
    )
    if config.training.projection_type in ["linear_violation", "linear_program", "convex_program"]:
        config.training.projection_kwargs["n_action"] = action_dim
        config.training.projection_kwargs["n_constraints"] = env.n_constraints
        projection_layer = ProjectionFactory.create_class(config.training.projection_type, config.training.projection_kwargs)
    else:
        projection_layer = None
    policy = ProjectionProbabilisticActor(
        module=actor,
        in_keys=["loc", "scale"],
        distribution_class=TruncatedNormal,
        distribution_kwargs={"low": env.action_spec.low[0], "high": env.action_spec.high[0]},
        # distribution_kwargs={"upscale":1.0},
        return_log_prob=True,
        projection_layer=projection_layer,
        # action_rescale_min=env.action_spec.low[0],
        # action_rescale_max=env.action_spec.high[0],
    )

    ## Main loop
    # Train the model
    if config.model.phase == "train":
        train(policy, critic, **config)
    # Test the model
    elif config.model.phase == "test":
            # todo: extract trained hyper-parameters
            # todo: check if two models can be re-loaded
            # todo: use validation function, add visualizations per episode
        raise NotImplementedError("Testing not implemented yet.")

def optimize_sac_loss(subdata, policy, critics, actor_optim, critic_optim, **kwargs):
    ## Hyperparameters
    gamma = kwargs["algorithm"]["gamma"]
    tau = kwargs["algorithm"]["tau"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]

    ## Unpack critics
    critic1 = critics["critic1"]
    critic2 = critics["critic2"]
    target_critic1 = critics["target_critic1"]
    target_critic2 = critics["target_critic2"]

    ## Loss optimization
    loss_out = {}
    # Critic loss calculation
    with torch.no_grad():
        # Next action
        next_policy_out = policy(subdata)
        next_action = next_policy_out["action"]
        next_log_prob = next_policy_out["sample_log_prob"].unsqueeze(-1)
        next_log_prob = torch.clamp(next_log_prob, -20, 2)  # Clip log_prob to avoid NaNs

        # Target value
        target_q1 = target_critic1(next_policy_out["observation"], next_action)
        target_q2 = target_critic2(next_policy_out["observation"], next_action)
        target_q_min = torch.min(target_q1, target_q2) - entropy_lambda * next_log_prob
        target_value = subdata["next", "reward"] + (1 - subdata["done"].float()) * gamma * target_q_min
        check_for_nans(target_value, "target_value")

    # Current value
    current_q1 = critic1(subdata["observation"], subdata["action"])
    current_q2 = critic2(subdata["observation"], subdata["action"])

    # Update critic
    loss_out["loss_critic"] = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)
    check_for_nans(loss_out["loss_critic"], "loss_critic")
    critic_optim.zero_grad()
    loss_out["loss_critic"].backward()
    loss_out["gn_critic1"] = torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_grad_norm)
    loss_out["gn_critic2"] = torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_grad_norm)
    check_for_nans(loss_out["loss_critic"], "loss_critic")
    check_for_nans(loss_out["gn_critic1"], "gn_critic1")
    check_for_nans(loss_out["gn_critic2"], "gn_critic2")
    critic_optim.step()

    # Soft update target critics
    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Compute Actor Loss
    policy_out = policy(subdata)
    policy_out["sample_log_prob"] = torch.clamp(policy_out["sample_log_prob"], -20, 2)  # Clip log_prob to avoid NaNs
    recursive_check_for_nans(policy_out)

    # Feasibility loss
    loss_out["loss_feasibility"], loss_out["mean_violation"] = compute_loss_feasibility(policy_out, policy_out["action"], feasibility_lambda, "sum")
    check_for_nans(loss_out["loss_feasibility"], "loss_feasibility")
    q1 = critic1(policy_out["observation"], policy_out["action"])
    q2 = critic2(policy_out["observation"], policy_out["action"])
    q_min = torch.min(q1, q2)
    check_for_nans(q_min, "q_min")

    # Update actor
    loss_out["loss_actor"] = (entropy_lambda * policy_out["sample_log_prob"].unsqueeze(-1) - q_min).mean() + loss_out["loss_feasibility"]
    actor_optim.zero_grad()
    loss_out["loss_actor"].backward()
    check_for_nans(loss_out["loss_actor"], "loss_actor")
    loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    check_for_nans(loss_out["gn_actor"], "gn_actor")
    actor_optim.step()
    return loss_out, policy_out

## Training
def train(policy, critic, device=torch.device("cuda"), **kwargs):
    # Hyperparameters
    batch_size = kwargs["model"]["batch_size"]
    mini_batch_size = int(kwargs["algorithm"]["mini_batch_size"] * batch_size)
    n_step = kwargs["algorithm"]["n_step"]
    num_epochs = kwargs["algorithm"]["ppo_epochs"]
    vf_lambda = kwargs["algorithm"]["vf_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]
    lr = kwargs["training"]["lr"]
    train_data_size = kwargs["training"]["train_data_size"]
    validation_episodes = kwargs["training"]["validation_episodes"]
    validation_freq = kwargs["training"]["validation_freq"]

    # Environment
    train_env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)

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
        critics = {"critic1": critic1, "critic2": critic2, "target_critic1": target_critic1, "target_critic2": target_critic2}

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
        train_env,
        policy,
        frames_per_batch=batch_size*n_step, # batch_size * steps_per_episode
        total_frames=train_data_size,
        split_trajs=False,
        device=device,
    )
    collector.set_seed(train_env.seed)
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
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, train_data_size)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optim, train_data_size)

    # Validation
    val_rewards = []
    decrease_count = 0
    patience = 2

    # Training loop
    for step, td in enumerate(collector):
        if kwargs["algorithm"]["type"] == "ppo_feas":
            advantage_module(td)
        replay_buffer.extend(td)
        for _ in range(batch_size // mini_batch_size):
            # Sample mini-batch (including actions, n-step returns, old log likelihoods, target_values)
            subdata = replay_buffer.sample(mini_batch_size).to(device)
            # Loss computation and backpropagation # todo: make sac into custom loss module
            if kwargs["algorithm"]["type"] == "sac":
                loss_out, policy_out = optimize_sac_loss(subdata, policy, critics, actor_optim, critic_optim, **kwargs)
            elif kwargs["algorithm"]["type"] == "ppo":
                raise NotImplementedError("PPO not implemented yet.")
            elif kwargs["algorithm"]["type"] == "ppo_feas":
                for _ in range(num_epochs):
                    loss_out = loss_module(subdata.to(device))
                    loss_out["total_loss"] = (loss_out["loss_objective"] + loss_out["loss_critic"]
                                              + loss_out["loss_entropy"] + loss_out["loss_feasibility"])

                    # Optimization: backward, grad clipping and optimization step
                    loss_out["total_loss"].backward()
                    loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(loss_module.parameters(), kwargs["algorithm"]["max_grad_norm"])
                    actor_optim.step()
                    actor_optim.zero_grad()

        # Log metrics
        train_performance = get_performance_metrics(subdata, td, train_env)
        log = {
            # Losses
            "total_loss": loss_out.get("total_loss", 0),
            "loss_actor": loss_out.get("loss_actor", loss_out.get("loss_objective")),
            "loss_critic":  loss_out["loss_critic"],
            "loss_feasibility":loss_out["loss_feasibility"],
            "mean_total_violation": loss_out["mean_violation"].sum(dim=(-2, -1)).mean().item(),
            "loss_entropy": loss_out.get("loss_entropy", 0),
            # Supporting metrics
            "step": step,
            "gn_actor": loss_out["gn_actor"].item(),
            "clip_fraction": loss_out.get("clip_fraction", 0),
            **train_performance,
        }
        pbar.update(1)
        # Log metrics
        pbar.set_description(
            # Loss, gn and rewards
            f"return: {log['return']: 4.4f}, "
            f"traj_return: {log['traj_return']: 4.4f}, "
            f"loss_actor:  {log['loss_actor']: 4.4f}, "
            f"loss_critic:  {log['loss_critic']: 4.4f}, "
            f"feasibility_loss: {log['loss_feasibility']: 4.4f}, "
            f"mean_violation: {log['mean_total_violation']: 4.4f}, "    
            # Prediction
            f"x: {log['x']: 4.4f}, "
            f"loc(x): {log['loc(x)']: 4.4f}, "
            f"scale(x): {log['scale(x)']: 4.4f}, "
            # Performance
            f"total_profit: {log['total_profit']: 4.4f}, "
            f"violation: {log['total_violation']: 4.4f}, "
        )

        # Validation step
        if (step + 1) % int(train_updates * validation_freq) == 0:
            validation_performance = validate_policy(train_env, policy, n_step=n_step, )
            log.update(validation_performance)
            val_rewards.append(validation_performance["validation"]["traj_return"])
            if early_stopping(val_rewards, patience):
                print(f"Early stopping at epoch {step} due to {patience} consecutive decreases in validation reward.")
                break

        # Update wandb and scheduler
        wandb.log(log)
        actor_scheduler.step()
        critic_scheduler.step()

    # todo: cleanup model saving
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model checkpoint with timestamp
    policy_save_path = f"saved_models/policy_{timestamp}.pth"
    critic_save_path = f"saved_models/critic_{timestamp}.pth"
    os.makedirs(os.path.dirname(policy_save_path), exist_ok=True)

    # Save the policy model
    torch.save(policy.state_dict(), policy_save_path)
    wandb.save(policy_save_path)

    # Save the critic model
    if kwargs["algorithm"]["type"] == "sac":
        torch.save(critic1.state_dict(), f"saved_models/critic1_{timestamp}.pth")
        torch.save(critic2.state_dict(), f"saved_models/critic2_{timestamp}.pth")
        torch.save(target_critic1.state_dict(), f"saved_models/target_critic1_{timestamp}.pth")
        torch.save(target_critic2.state_dict(), f"saved_models/target_critic2_{timestamp}.pth")
        wandb.save(f"saved_models/critic1_{timestamp}.pth")
        wandb.save(f"saved_models/critic2_{timestamp}.pth")
        wandb.save(f"saved_models/target_critic1_{timestamp}.pth")
        wandb.save(f"saved_models/target_critic2_{timestamp}.pth")
    else:
        torch.save(critic.state_dict(), critic_save_path)
        wandb.save(critic_save_path)

    # Close environments
    train_env.close()

## Validation
def validate_policy(env: EnvBase, policy_module: ProbabilisticActor, num_episodes: int = 10, n_step: int = 100,):
    """Validate the policy using the environment."""
    # Perform a rollout to evaluate the policy
    with torch.no_grad():
        trajectory = env.rollout(policy=policy_module, max_steps=n_step, auto_reset=True)
    val_metrics = get_performance_metrics(trajectory, trajectory, env)
    return {"validation": val_metrics}

def get_performance_metrics(subdata, td, env):
    """Compute performance metrics for the policy."""
    return {# Return
            "return": subdata['next', 'reward'].mean().item(),
            "traj_return": subdata['next', 'reward'].sum(dim=(-2, -1)).mean().item(),

            # Prediction
            "x": subdata['action'].mean().item(),
            "loc(x)": subdata['loc'].mean().item(),
            "scale(x)": subdata['scale'].mean().item(),

            # Constraints
            "total_violation": subdata['violation'].sum(dim=(-2,-1)).mean().item(),
            "demand_violation": subdata['violation'][...,0].sum(dim=(1)).mean().item(),
            "capacity_violation": subdata['violation'][...,1:-4].sum(dim=(1)).mean().item(),
            "LCG_violation": subdata['violation'][..., env.next_port_mask, -4:-2].sum(dim=(1,2)).mean().item(),
            "VCG_violation": subdata['violation'][..., env.next_port_mask, -2:].sum(dim=(1,2)).mean().item(),

            # Environment
            "total_revenue": subdata["revenue"].sum(dim=(-2,-1)).mean().item(),
            "total_cost": subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_profit": subdata["revenue"].sum(dim=(-2,-1)).mean().item() -
                            subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_loaded": subdata["action"].sum(dim=(-2,-1)).mean().item(),
            "total_demand":subdata['realized_demand'][:,0,:].sum(dim=-1).mean(),
            "total_e[x]_demand": td['init_expected_demand'][:, 0, :].sum(dim=-1).mean(),
            "mean_std[x]_demand": subdata['std_demand'][:, 0, :].std(dim=-1).mean(),
        }

## Early stopping
def early_stopping(val_rewards, patience=2):
    """
    Check for early stopping based on consecutive decreases in validation rewards.

    Args:
        val_rewards (list): A list of validation rewards.
        patience (int): Number of consecutive decreases allowed before triggering early stopping.

    Returns:
        bool: True if early stopping condition is met, otherwise False.
    """
    # Track the number of consecutive decreases
    decrease_count = 0

    for i in range(1, len(val_rewards)):
        if val_rewards[i] < val_rewards[i - 1]:
            decrease_count += 1
            if decrease_count >= patience:
                return True
        else:
            decrease_count = 0  # Reset if the reward improves or stays the same

    return False

def recursive_check_for_nans(td, parent_key=""):
    """Recursive check for NaNs and Infs in e.g. TensorDicts and raise an error if found."""
    for key, value in td.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, torch.Tensor):
            check_for_nans(value, full_key)
        elif isinstance(value, TensorDict) or  isinstance(value, Dict):
            # Recursively check nested TensorDicts
            recursive_check_for_nans(value, full_key)

def check_for_nans(tensor, name):
    """Check for NaNs and Infs in a tensor and raise an error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

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