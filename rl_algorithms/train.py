from datetime import datetime
import os
import copy
import wandb
import tqdm
import yaml
from dotmap import DotMap

# Torch
import torch

# TorchRL
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Custom code
from rl_algorithms.utils import make_env
from rl_algorithms.loss import FeasibilityClipPPOLoss, FeasibilitySACLoss

# Functions
def convert_to_dict(obj):
    """Recursively convert DotMap or other custom objects to standard Python dictionaries."""
    if isinstance(obj, DotMap):
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):  # Handle nested dictionaries
        return {key: convert_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, list):  # Handle lists containing DotMaps or dicts
        return [convert_to_dict(item) for item in obj]
    return obj  # Return primitive data types as-is

def run_training(policy, critic, device=torch.device("cuda"), **kwargs):
    """Train the policy using the specified algorithm."""
    # Algorithm hyperparameters
    lr = kwargs["training"]["lr"]
    batch_size = kwargs["model"]["batch_size"]
    mini_batch_size = int(kwargs["algorithm"]["mini_batch_size"] * batch_size)
    n_step = kwargs["algorithm"]["n_step"]
    num_epochs = kwargs["algorithm"]["ppo_epochs"]
    gamma = kwargs["algorithm"]["gamma"]
    gae_lambda = kwargs["algorithm"]["gae_lambda"]
    # Loss hyperparameters
    vf_lambda = kwargs["algorithm"]["vf_lambda"]
    feasibility_lambda = kwargs["algorithm"]["feasibility_lambda"]
    entropy_lambda = kwargs["algorithm"]["entropy_lambda"]
    clip_epsilon = kwargs["algorithm"]["clip_range"]
    max_grad_norm = kwargs["algorithm"]["max_grad_norm"]
    tau = kwargs["algorithm"]["tau"]
    # Training hyperparameters
    train_data_size = kwargs["training"]["train_data_size"]
    validation_freq = kwargs["training"]["validation_freq"]
    validation_patience = kwargs["training"]["validation_patience"]

    # Environment
    train_env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)

    # Optimizer, loss module, data collector, and scheduler
    advantage_module = GAE(gamma=gamma, lmbda=gae_lambda, value_network=critic, average_gae=True)
    if kwargs["algorithm"]["type"] == "sac":
        loss_module = FeasibilitySACLoss(
            actor_network=policy,
            qvalue_network=critic,
            separate_losses=True,
            fixed_alpha=False,
            # min_alpha=1e-2, #[1e-2, 1e-3]
            # max_alpha=1.0, #[1.0, 10]
        )
    elif kwargs["algorithm"]["type"] == "ppo":
        loss_module = FeasibilityClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=clip_epsilon,
            entropy_bonus=bool(entropy_lambda),
            entropy_coef=entropy_lambda,
            critic_coef=vf_lambda,
            loss_critic_type="smooth_l1",
            normalize_advantage=True,
        )
    elif kwargs["algorithm"]["type"] == "ddpg":
        # Create the DDPG loss module
        loss_module = DDPGLoss(
            actor_network=policy,
            value_network=critic,
            delay_actor=True,
            delay_value=True,
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
    critic_optim = torch.optim.Adam(loss_module.qvalue_network_params.parameters(), lr=lr)
    if not loss_module.fixed_alpha:
        alpha_optim = torch.optim.Adam([loss_module.log_alpha], lr=lr)

    train_updates = train_data_size // (batch_size * n_step)
    pbar = tqdm.tqdm(range(train_updates))
    actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optim, train_data_size)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(critic_optim, train_data_size)

    # Validation
    val_rewards = []
    policy.train()
    # torch.autograd.set_detect_anomaly(True)
    # Training loop
    for step, td in enumerate(collector):
        if kwargs["algorithm"]["type"] == "ppo_feas":
            advantage_module(td)
        replay_buffer.extend(td)
        for _ in range(batch_size // mini_batch_size):
            # Sample mini-batch (including actions, n-step returns, old log likelihoods, target_values)
            subdata = replay_buffer.sample(mini_batch_size).to(device)
            # Loss computation and backpropagation
            if kwargs["algorithm"]["type"] == "sac":
                # Loss computation
                loss_out = loss_module(subdata.to(device))

                # Critic Update
                loss_out["loss_qvalue"].backward()
                qvalue_params = loss_module.qvalue_network_params.flatten_keys().values()
                loss_out["gn_critic"] = torch.nn.utils.clip_grad_norm_(qvalue_params, max_grad_norm)
                critic_optim.step()
                # Soft update target critics
                with torch.no_grad():
                    soft_update(loss_module.target_qvalue_network_params, loss_module.qvalue_network_params, tau)
                critic_optim.zero_grad()

                # Actor Update
                loss_out["loss_actor"] = loss_out["loss_actor"].clone() + feasibility_lambda * loss_out["loss_feasibility"]
                loss_out["loss_actor"].backward()
                loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                actor_optim.step()
                actor_optim.zero_grad()

                # Alpha Update
                if not loss_module.fixed_alpha:
                    loss_out["loss_alpha"].backward()
                    alpha_optim.step()
                    alpha_optim.zero_grad()
            elif kwargs["algorithm"]["type"] == "ppo":
                for _ in range(num_epochs):
                    loss_out = loss_module(subdata.to(device))
                    loss_out["total_loss"] = (loss_out["loss_objective"] + loss_out["loss_critic"]
                                              + loss_out["loss_entropy"] + loss_out["loss_feasibility"])

                    # Optimization: backward, grad clipping and optimization step
                    loss_out["total_loss"].backward()
                    loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), kwargs["algorithm"]["max_grad_norm"])
                    actor_optim.step()
                    actor_optim.zero_grad()

        # Log metrics
        train_performance = get_performance_metrics(subdata, td, train_env)
        log = {
            # Losses
            "total_loss": loss_out.get("total_loss", 0),
            "loss_actor": loss_out.get("loss_actor", 0) or loss_out.get("loss_objective", 0),
            "loss_critic": loss_out.get("loss_qvalue", 0) or loss_out.get("loss_critic", 0),
            "loss_feasibility":loss_out.get("loss_feasibility", 0),
            "violation": loss_out.get("violation", 0),
            "loss_entropy": loss_out.get("loss_alpha", 0) or loss_out.get("loss_entropy", 0),
            # Supporting metrics
            "step": step,
            "gn_actor": loss_out["gn_actor"].item(),
            "gn_critic": loss_out.get("gn_critic", 0),
            "clip_fraction": loss_out.get("clip_fraction", 0),
            **train_performance,
        }
        log["mean_total_violation"] = log["violation"].sum(dim=(-2, -1)).mean().item() if log["violation"].dim() > 1 else 0
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
            policy.eval()
            # todo: check if projection is used/needed here
            validation_performance = validate_policy(train_env, policy, n_step=n_step, )
            log.update(validation_performance)
            val_rewards.append(validation_performance["validation"]["traj_return"])
            if early_stopping_val(val_rewards, validation_patience):
                print(f"Early stopping at epoch {step} due to {validation_patience} consecutive decreases in validation reward.")
                break
            # Save models (create a new directory for each validation)
            save_models(policy, loss_module, critic, kwargs["algorithm"]["type"], kwargs, save_dir="saved_models/validation")
            policy.train()

        # Early stopping due to divergence
        # check total_loss with ppo, loss_actor with sac
        check_loss = log["loss_actor"] if kwargs["algorithm"]["type"] == "sac" else log["total_loss"]
        if early_stopping_divergence(check_loss):
            print(f"Early stopping at epoch {step} due to loss divergence.")
            break

        # Update wandb and scheduler
        wandb.log(log)
        actor_scheduler.step()
        critic_scheduler.step()

    # Save models and close environment
    save_models(policy, loss_module, critic, kwargs["algorithm"]["type"], kwargs)
    train_env.close()

def get_performance_metrics(subdata, td, env):
    """Compute performance metrics for the policy."""
    return {# Return
            "return": subdata["next", "reward"].mean().item(),
            "traj_return": subdata["next", "reward"].sum(dim=(-2, -1)).mean().item(),

            # Prediction
            "x": subdata["action"].mean().item(),
            "loc(x)": subdata["loc"].mean().item(),
            "scale(x)": subdata["scale"].mean().item(),

            # Constraints
            "total_violation": subdata["violation"].sum(dim=(-2,-1)).mean().item(),
            "demand_violation": subdata["violation"][...,0].sum(dim=(1)).mean().item(),
            "capacity_violation": subdata["violation"][...,1:-4].sum(dim=(1)).mean().item(),
            "LCG_violation": subdata["violation"][..., env.next_port_mask, -4:-2].sum(dim=(1,2)).mean().item(),
            "VCG_violation": subdata["violation"][..., env.next_port_mask, -2:].sum(dim=(1,2)).mean().item(),

            # Environment
            "total_revenue": subdata["revenue"].sum(dim=(-2,-1)).mean().item(),
            "total_cost": subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_profit": subdata["revenue"].sum(dim=(-2,-1)).mean().item() -
                            subdata["cost"].sum(dim=(-2,-1)).mean().item(),
            "total_loaded": subdata["action"].sum(dim=(-2,-1)).mean().item(),
            "total_demand":subdata["observation", "realized_demand"][:,0,:].sum(dim=-1).mean(),
            "total_e[x]_demand": td["observation", "init_expected_demand"][:, 0, :].sum(dim=-1).mean(),
            "mean_std[x]_demand": subdata["observation", "std_demand"][:, 0, :].std(dim=-1).mean(),
        }

def validate_policy(env: EnvBase, policy_module: ProbabilisticActor, num_episodes: int = 10, n_step: int = 100,):
    """Validate the policy using the environment."""
    # Perform a rollout to evaluate the policy
    with torch.no_grad():
        trajectory = env.rollout(policy=policy_module, max_steps=n_step, auto_reset=True)
    val_metrics = get_performance_metrics(trajectory, trajectory, env)
    return {"validation": val_metrics}

def early_stopping_val(val_rewards, patience=5):
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

def early_stopping_divergence(loss, threshold=1e6):
    """
    Check for early stopping based on a threshold for the loss value.

    Args:
        loss (float): The loss value to check.
        threshold (float): The threshold value for the loss.

    Returns:
        bool: True if early stopping condition is met, otherwise False.
    """
    # Check if nan or inf in the loss
    if torch.isnan(loss) or torch.isinf(loss):
        return True

    # Check if the loss exceeds the threshold
    if loss > threshold:
        return True
    return False

def soft_update(target_params, source_params, tau):
    """Soft update the target parameters using the source parameters."""
    for target, source in zip(target_params.flatten_keys().values(), source_params.flatten_keys().values()):
        target.copy_(tau * source + (1.0 - tau) * target)

def save_models(policy, loss_module, critic, algorithm_type, kwargs_train, save_dir="saved_models"):
    """
    Save the policy and critic models with a timestamped directory structure.

    Args:
        policy (torch.nn.Module): The policy model to save.
        loss_module: Loss module containing Q-value networks and target Q-value networks (for SAC).
        critic (torch.nn.Module): The critic model to save (for non-SAC algorithms).
        algorithm_type (str): The type of algorithm (e.g., "sac").
        save_dir (str): Base directory for saving models.
    """
    # Generate a timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Save the policy model
    policy_save_path = os.path.join(save_path, "policy.pth")
    torch.save(policy.state_dict(), policy_save_path)
    wandb.save(policy_save_path)

    # Save the critic model(s)
    if algorithm_type == "sac":
        critic_paths = {
            "critic1": os.path.join(save_path, "critic1.pth"),
            "critic2": os.path.join(save_path, "critic2.pth"),
            "target_critic1": os.path.join(save_path, "target_critic1.pth"),
            "target_critic2": os.path.join(save_path, "target_critic2.pth"),
        }

        torch.save(loss_module.qvalue_network_params[0].state_dict(), critic_paths["critic1"])
        torch.save(loss_module.qvalue_network_params[1].state_dict(), critic_paths["critic2"])
        torch.save(loss_module.target_qvalue_network_params[0].state_dict(), critic_paths["target_critic1"])
        torch.save(loss_module.target_qvalue_network_params[1].state_dict(), critic_paths["target_critic2"])

        # Log critic models to wandb
        for path in critic_paths.values():
            wandb.save(path)
    else:
        critic_save_path = os.path.join(save_path, "critic.pth")
        torch.save(critic.state_dict(), critic_save_path)
        wandb.save(critic_save_path)

    # Save the configuration to a YAML file
    config_save_path = os.path.join(save_path, "config.yaml")
    cleaned_config = convert_to_dict(kwargs_train)  # Convert DotMap to dictionary
    with open(config_save_path, "w") as yaml_file:
        yaml.dump(cleaned_config, yaml_file, default_flow_style=False)
    wandb.save(config_save_path)