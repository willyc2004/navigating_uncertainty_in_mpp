from datetime import datetime
import os
import copy
import wandb
import tqdm

# Torch
import torch

# TorchRL
from torchrl.envs import EnvBase
from torchrl.modules import ProbabilisticActor
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Custom code
from rl_algorithms.utils import make_env
from rl_algorithms.loss import FeasibilityClipPPOLoss, optimize_sac_loss

# Training
def train(policy, critic, device=torch.device("cuda"), **kwargs):
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
    # Training hyperparameters
    train_data_size = kwargs["training"]["train_data_size"]
    validation_freq = kwargs["training"]["validation_freq"]
    validation_patience = kwargs["training"]["validation_patience"]

    # Environment
    train_env = make_env(env_kwargs=kwargs["env"], batch_size=[batch_size], device=device)

    # Optimizer, loss module, data collector, and scheduler
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
        # todo: make feasibility_sac_loss_module
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
    policy.train()
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
                raise NotImplementedError("PPO without feasibility not implemented yet.")
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
            policy.eval()
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

# Get performance metrics
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
            "total_demand":subdata['state', 'realized_demand'][:,0,:].sum(dim=-1).mean(),
            "total_e[x]_demand": td['state', 'init_expected_demand'][:, 0, :].sum(dim=-1).mean(),
            "mean_std[x]_demand": subdata['state', 'std_demand'][:, 0, :].std(dim=-1).mean(),
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
