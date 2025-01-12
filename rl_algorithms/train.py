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
from rl_algorithms.loss import FeasibilityClipPPOLoss, FeasibilitySACLoss, CustomSACLoss

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

# Training
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
        critic1, critic2 = critic
        target_critic1 = copy.deepcopy(critic1).to(device)
        target_critic2 = copy.deepcopy(critic2).to(device)
        loss_module = FeasibilitySACLoss(
            actor_network=policy,
            qvalue_network=critic,  # List of two Q-networks
            fixed_alpha=True,
            min_alpha=1e-2, #[1e-2, 1e-3]
            max_alpha=1.0, #[1.0, 10]
        )
        # loss_module = CustomSACLoss(
        #     actor=policy,
        #     critics=[critic1, critic2],  # List of two Q-networks
        #     target_critics=[target_critic1, target_critic2],  # List of two target Q-networks
        #     value_network=None,
        # )
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
    if kwargs["algorithm"]["type"] == "sac":
        critic_optim = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=lr)
        alpha = torch.nn.Parameter(torch.tensor(0.0, device=device))
        alpha_optim = torch.optim.Adam([alpha], lr=lr)
    else:
        critic_optim = torch.optim.Adam(critic.parameters(), lr=lr)
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
                critic_optim.zero_grad()
                loss_out["loss_qvalue"].backward()
                loss_out["gn_critic1"] = torch.nn.utils.clip_grad_norm_(critic1.parameters(), max_grad_norm)
                loss_out["gn_critic2"] = torch.nn.utils.clip_grad_norm_(critic2.parameters(), max_grad_norm)
                critic_optim.step()

                # Actor Update
                actor_optim.zero_grad()
                loss_out["loss_actor"] += feasibility_lambda * loss_out["loss_feasibility"]
                loss_out["loss_actor"].backward()
                loss_out["gn_actor"] = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                actor_optim.step()
                #
                # # Alpha Update
                # alpha_optim.zero_grad()
                # loss_out["loss_alpha"] = entropy_lambda * loss_out["loss_alpha"]
                # loss_out["loss_alpha"].backward()
                # alpha_optim.step()

                # Soft update target critics
                for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for name, param in critic[0].named_parameters():
                    if param.grad is None:
                        print(f"No gradient for {name}")
                    else:
                        print(f"{name} grad norm: {param.grad.norm().item()}")


                print(f"\nCritic Loss: {loss_out['loss_qvalue'].item()}, "
                      f"Actor Loss: {loss_out['loss_actor'].item()}, "
                      f"Alpha Loss: {loss_out['loss_alpha'].item()}, "
                      f"Grad Norms - Critic1: {loss_out['gn_critic1']}, "
                      f"Critic2: {loss_out['gn_critic2']}, Actor: {loss_out['gn_actor']}")
                breakpoint()
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
            "mean_total_violation": loss_out.get("violation", torch.tensor([0.], device=device).unsqueeze(-1)).sum(dim=(-2, -1)).mean().item()
                                    or loss_out.get("mean_violation", torch.tensor([0.], device=device).unsqueeze(-1)).sum(dim=(-2, -1)).mean().item(),
            "loss_entropy": loss_out.get("loss_alpha", 0) or loss_out.get("loss_entropy", 0),
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
            policy.eval()
            # todo: check if projection is used/needed here
            validation_performance = validate_policy(train_env, policy, n_step=n_step, )
            log.update(validation_performance)
            val_rewards.append(validation_performance["validation"]["traj_return"])
            if early_stopping(val_rewards, validation_patience):
                print(f"Early stopping at epoch {step} due to {validation_patience} consecutive decreases in validation reward.")
                break
            policy.train()

        # Update wandb and scheduler
        wandb.log(log)
        actor_scheduler.step()
        critic_scheduler.step()


    # todo: make clean function
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model checkpoint with timestamp
    save_path = f"saved_models/{timestamp}/"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the policy model
    policy_save_path = os.path.join(save_path, "policy.pth")
    torch.save(policy.state_dict(), policy_save_path)
    wandb.save(policy_save_path)

    # Save the critic model
    if kwargs["algorithm"]["type"] == "sac":
        critic_paths = {
            "critic1": os.path.join(save_path, "critic1.pth"),
            "critic2": os.path.join(save_path, "critic2.pth"),
            "target_critic1": os.path.join(save_path, "target_critic1.pth"),
            "target_critic2": os.path.join(save_path, "target_critic2.pth"),
        }
        torch.save(critic1.state_dict(), critic_paths["critic1"])
        torch.save(critic2.state_dict(), critic_paths["critic2"])
        torch.save(target_critic1.state_dict(), critic_paths["target_critic1"])
        torch.save(target_critic2.state_dict(), critic_paths["target_critic2"])

        # Log to wandb
        for path in critic_paths.values():
            wandb.save(path)
    else:
        critic_save_path = os.path.join(save_path, "critic.pth")
        torch.save(critic.state_dict(), critic_save_path)
        wandb.save(critic_save_path)

    # Save the configuration to a YAML file
    config_save_path = os.path.join(save_path, "config.yaml")
    cleaned_config = convert_to_dict(kwargs) # Convert DotMap to dictionary
    with open(config_save_path, "w") as yaml_file:
        yaml.dump(cleaned_config, yaml_file, default_flow_style=False)
    wandb.save(config_save_path)

    # Close environments
    train_env.close()

# Get performance metrics
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

# Validation
def validate_policy(env: EnvBase, policy_module: ProbabilisticActor, num_episodes: int = 10, n_step: int = 100,):
    """Validate the policy using the environment."""
    # Perform a rollout to evaluate the policy
    with torch.no_grad():
        trajectory = env.rollout(policy=policy_module, max_steps=n_step, auto_reset=True)
    val_metrics = get_performance_metrics(trajectory, trajectory, env)
    return {"validation": val_metrics}

# Early stopping
def early_stopping(val_rewards, patience=5):
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
