from typing import Any, Union
import copy
from tensordict import TensorDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchrl.objectives.value.functional import generalized_advantage_estimate
from torch.utils.data import BatchSampler, SubsetRandomSampler
import os

# Enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from torchrl.data.replay_buffers import (
    LazyMemmapStorage,
    ListStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.models.rl.common.base import RL4COLitModule
from rl4co.models.rl.common.critic import CriticNetwork, create_critic_from_actor
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class Memory:
    def __init__(self, batch_size, max_steps, env, device="cuda"):
        self.tds = []
        self.actions = torch.zeros((*batch_size, max_steps, env.D * env.B), device=device)
        self.logprobs = torch.zeros((*batch_size, max_steps, env.D * env.B), device=device)
        self.rewards = torch.zeros((*batch_size, max_steps, 1), device=device)
        self.values = torch.zeros((*batch_size, max_steps, 1), device=device)
        self.profit = torch.zeros((*batch_size, max_steps, 1), device=device)

    def clear_memory(self):
        del self.tds[:]
        self.actions = self.actions.new_zeros(self.actions.size())
        self.logprobs = self.logprobs.new_zeros(self.logprobs.size())
        self.rewards = self.rewards.new_zeros(self.rewards.size())
        self.values = self.values.new_zeros(self.values.size())
        self.profit = self.profit.new_zeros(self.profit.size())

class ObservationNormalizer(nn.Module):
    def __init__(self, obs_spec, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.epsilon = epsilon
        self.momentum = momentum

        # Ensure mean and std shapes match the non-batch dimensions of obs_spec
        self.mean = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(spec.shape[1:], device=spec.device), requires_grad=False)
            for key, spec in obs_spec.items()
        })
        self.std = nn.ParameterDict({
            key: nn.Parameter(torch.ones(spec.shape[1:], device=spec.device), requires_grad=False)
            for key, spec in obs_spec.items()
        })

    def forward(self, obs):
        # Normalize each entry in the TensorDict
        normalized_obs = {
            key: (obs[key] - self.mean[key]) / (self.std[key] + self.epsilon)
            for key in obs.keys()
        }
        # give example

        return TensorDict(normalized_obs, batch_size=obs.batch_size, device=obs.device)

    def update_stats(self, new_obs):
        # Update running mean and std for each entry in the TensorDict along the batch dimension
        for key, value in new_obs.items():
            # Move mean and std to the same device as new_obs for compatibility
            self.mean[key].data = self.mean[key].data.to(value.device)
            self.std[key].data = self.std[key].data.to(value.device)

            # Calculate batch mean and std along the batch dimension
            batch_mean = value.mean(dim=0)  # Calculate mean across batch
            batch_std = value.std(dim=0)  # Calculate std across batch

            if torch.isnan(batch_mean).any() or torch.isnan(batch_std).any():
                print(f"Warning: NaNs detected in batch statistics for '{key}'. Replacing with zero mean and min_std.")
                batch_mean = torch.nan_to_num(batch_mean, nan=0.0)
                batch_std = torch.nan_to_num(batch_std, nan=self.epsilon)
                breakpoint()

            # Update running statistics with momentum for stability
            self.mean[key].data = self.momentum * batch_mean + (1 - self.momentum) * self.mean[key].data
            self.std[key].data = self.momentum * batch_std + (1 - self.momentum) * self.std[key].data

class Projection_Nstep_PPO(RL4COLitModule):
    """
    An implementation of the n-step Proximal Policy Optimization (PPO) algorithm (https://arxiv.org/abs/2110.02544)
    is presented for training improvement models.
    """
    def __init__(
        self,
        env: RL4COEnvBase,
        policy: nn.Module,
        critic: CriticNetwork = None,
        critic_kwargs: dict = {},
        update_timestep: int = 1,
        clip_range: float = 0.1,  # epsilon of PPO
        ppo_epochs: int = 3,  # inner epoch, K
        vf_lambda: float = 0.5,  # lambda of Value function fitting
        entropy_lambda: float = 0.0,  # lambda of entropy bonus
        feasibility_lambda: float = 1.0,  # lambda of feasibility loss
        demand_lambda: float = 1.0,  # lambda of demand violations
        stability_lambda: float = 1.0,  # lambda of stability violations
        adaptive_feasibility_lambda: bool = False,  # whether to adapt feasibility lambda
        projection_lambda: float = 1.0,  # lambda of projection loss
        normalize_adv: bool = False,  # whether to normalize advantage
        normalize_return: bool = False,  # whether to normalize return
        max_grad_norm: float = 0.5,  # max gradient norm
        kl_threshold: float = 0.03,  # KL threshold
        kl_penalty_lambda: float = 1.0,  # KL penalty coefficient
        batch_size: int = 256,  # batch size
        mini_batch_size: Union[int, float] = 0.25,  # mini batch size,
        buffer_size: int = 100_000,
        buffer_storage_device: str = "gpu",
        metrics: dict = {
            "train": ["loss", "surrogate_loss", "value_loss", "entropy",
                      "feasibility_loss", "projection_loss",
                      "return", "adv", "value_pred", "ratios", "clipped_ratios",
                      "episodic_reward", "total_loaded", "violations",
                        "proj_mean_logits", "std_logits", "profit"
                      ],
                      # "costs", "revenue",
                        # "ll", "old_ll",
                        # "action", "logprobs", "logprobs_old",]
        },
        gamma: float = 0.99,  # gamma
        gae_lambda: float = 0.95,  # lambda of GAE
        n_step: float = 72,  # n-step for n-step PPO
        T_train: int = 72,  # the maximum inference T used for training
        T_test: int = 72,  # the maximum inference T used for test
        lr: float = 1e-4,  # learning rate of policy network
        lr_scheduler=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs: dict = {
            "gamma": 0.985,  # the learning decay per epoch,
        },
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_monitor=None,
        **kwargs,
    ):
        self.projection_type = kwargs.pop("projection_type", None)  # pop before passing to super
        self.projection_kwargs = kwargs.pop("projection_kwargs", None)  # pop before passing to super
        super().__init__(
            env,
            policy,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            optimizer_kwargs={"lr": lr, "weight_decay": 0.01},
            **kwargs,
        )
        self.automatic_optimization = False  # PPO uses custom optimization routine
        if critic is None:
            log.info("Creating critic network for {}".format(env.name))
            critic = create_critic_from_actor(policy, **critic_kwargs)
        self.critic = critic
        self.ppo_cfg = {
            "clip_range": clip_range,
            "ppo_epochs": ppo_epochs,
            "vf_lambda": vf_lambda,
            "normalize_adv": normalize_adv,
            "normalize_return": normalize_return,
            "max_grad_norm": max_grad_norm,
            "mini_batch_size": mini_batch_size,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "n_step": n_step,
            "T_train": T_train,
            "T_test": T_test,
            "lr": lr,
            "entropy_lambda": entropy_lambda,
            "adaptive_feasibility_lambda":adaptive_feasibility_lambda,
            "feasibility_lambda": feasibility_lambda,
            "projection_lambda": projection_lambda,
            "update_timestep": update_timestep,
            "kl_threshold": kl_threshold,
            "kl_penalty_lambda": kl_penalty_lambda,
        }
        self.mini_batch_size = (
            int(self.ppo_cfg["mini_batch_size"] * batch_size)
            if self.ppo_cfg["mini_batch_size"] < 1
            else self.ppo_cfg["mini_batch_size"]
        )

        self.lambda_violations = torch.tensor([demand_lambda] + [stability_lambda] * env.n_stability,
                                              device='cuda', dtype=torch.float32)

        # Normalization
        self.obs_norm = ObservationNormalizer(self.env.obs_spec)
        initialize_stats_from_rollouts(env, self.obs_norm, batch_size=mini_batch_size)
        self.return_mean = 0
        self.return_var = 0
        self.return_count = 0
        self.epsilon = 1e-4

    def configure_optimizers(self):
        parameters = list(self.policy.parameters()) + list(self.critic.parameters())
        return super().configure_optimizers(parameters)

    def on_train_epoch_end(self):
        """
        Learning rate scheduler and CL scheduler
        """
        # Learning rate scheduler
        sch = self.lr_schedulers()
        sch.step()

    def update_running_return_stats(self, returns):
        """
        Update running mean and variance for return normalization.
        """
        batch_count = returns.numel()
        batch_mean = returns.mean()
        batch_var = returns.var(unbiased=False)

        # Welford's algorithm for numerically stable mean and variance
        delta = batch_mean - self.return_mean
        total_count = self.return_count + batch_count
        self.return_mean += delta * batch_count / total_count
        self.return_var += batch_var * batch_count + delta ** 2 * (batch_count * self.return_count / total_count)
        self.return_count = total_count

    def normalize_returns(self, returns):
        """
        Normalize returns using running mean and variance.
        """
        if self.return_count < 2:  # Avoid division by zero in early stages
            return returns
        running_std = torch.sqrt(self.return_var / (self.return_count - 1 + self.epsilon))
        return (returns - self.return_mean) / (running_std + self.epsilon)

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        if phase != "train":
            td = self.env.reset(batch)
            out = self.policy.generate(td, env=self.env, phase=phase, return_feasibility=True,
                                       projection_type=self.projection_type, projection_kwargs=self.projection_kwargs)
        else:
            memory = Memory(batch.batch_size, self.ppo_cfg["n_step"],self.env)
            td = self.env.reset(batch)
            out = self.update(td, memory, phase, mini_batch_size=self.mini_batch_size)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def update(self, td, memory, phase, tolerance=1e-4, alpha=0.5, mini_batch_size=256):
        assert (
                self.ppo_cfg["T_train"] % self.ppo_cfg["n_step"] == 0
        ), "T_train should be divided by n_step with no remainder"
        list_metrics = []

        # Gather n_steps in memory
        with torch.no_grad():
            memory.clear_memory()
            for i in range(self.ppo_cfg["n_step"]):
                memory.tds.append(td.clone())
                td = self.policy.act(memory.tds[i], self.env, phase=phase)
                memory.values[:, i] = self.critic(memory.tds[i]).view(-1, 1)
                td = self.env.step(td.clone())["next"]
                memory.actions[:, i] = td["action"]
                memory.logprobs[:, i] = td["logprobs"]
                memory.rewards[:, i] = td["reward"].view(-1, 1)
                memory.profit[:, i] = td["profit"].view(-1, 1)

        # Run PPO inner epochs with minibatching
        for k in range(self.ppo_cfg["ppo_epochs"]):
            indices = torch.arange(td.batch_size[0])

            # Split indices into mini-batches for efficient batch processing
            for mini_batch_indices in BatchSampler(SubsetRandomSampler(indices), mini_batch_size, drop_last=True):
                mini_batch_indices = torch.tensor(mini_batch_indices, device=td.device)
                mini_batch_tds = torch.stack(memory.tds, dim=-1)[mini_batch_indices]
                mini_batch_actions = memory.actions[mini_batch_indices]

                # Perform a single forward pass for the n-step batch
                out = self.policy.evaluate(
                    mini_batch_tds,
                    action=mini_batch_actions,
                    env=self.env,
                    phase=phase,
                    return_actions=False,
                )

                # Store values directly in the memory object
                value_preds = self.critic(mini_batch_tds)
                dones = out["done"]

                # Compute advantages and returns
                old_values = memory.values[mini_batch_indices]
                rewards = memory.rewards[mini_batch_indices]
                adv, returns = generalized_advantage_estimate(
                    gamma=self.ppo_cfg["gamma"],
                    lmbda=self.ppo_cfg["gae_lambda"],
                    state_value=old_values,
                    next_state_value=value_preds,
                    reward=rewards,
                    done=dones.bool(),
                )

                # Normalize advantages and returns if enabled
                adv = self._normalize_if_enabled(adv, self.ppo_cfg["normalize_adv"])
                returns = self._normalize_if_enabled(returns, self.ppo_cfg["normalize_return"])

                # Compute losses and metrics
                loss, metrics = self._compute_losses(
                    adv, returns, out, memory, td, mini_batch_indices, self.lambda_violations, tolerance, alpha)

                # Perform the backward pass and optimization
                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                if self.ppo_cfg["max_grad_norm"] is not None:
                    self.clip_gradients(
                        opt,
                        gradient_clip_val=self.ppo_cfg["max_grad_norm"],
                        gradient_clip_algorithm="norm",
                    )
                opt.step()
                list_metrics.append(metrics)

        return self._aggregate_metrics(list_metrics)

    def _normalize_if_enabled(self, tensor, enabled):
        """Normalize tensors if enabled in configuration"""
        return (tensor - tensor.mean()) / (tensor.std() + 1e-8) if enabled else tensor

    def _compute_losses(self, adv, returns, out, memory, td,
                        mini_batch_indices, lambda_values, tolerance, alpha):
        """Compute the PPO loss, value loss, entropy loss, feasibility loss, and projection loss."""
        # Extract from memory/out
        old_ll = memory.logprobs[mini_batch_indices]
        value_preds = memory.values[mini_batch_indices]
        ll = out["logprobs"]
        entropy = out["entropy"]
        mean_logits = out["mean_logits"]
        proj_mean_logits = out["proj_mean_logits"]
        lhs_A = out["lhs_A"]
        rhs = out["rhs"]

        # Compute the ratios for PPO clipping
        log_ratios = ll - old_ll.detach()
        ratios = torch.exp(log_ratios)
        clipped_ratios = torch.clamp(ratios, 1 - self.ppo_cfg["clip_range"], 1 + self.ppo_cfg["clip_range"])
        surrogate_loss = -torch.min(ratios * adv, clipped_ratios * adv).mean()

        # Compute the value and entropy loss
        value_loss = F.huber_loss(value_preds.detach(), returns, reduction="mean")

        # Feasibility and projection losses
        lhs = (lhs_A * proj_mean_logits.unsqueeze(-2)).sum(dim=-1)
        violation = torch.clamp(lhs - rhs, min=0)
        if self.ppo_cfg["adaptive_feasibility_lambda"]:
            lambda_values = torch.where(
                violation > tolerance,
                lambda_values * (1 + alpha * violation),
                lambda_values,
            )
        feasibility_loss = F.mse_loss(lambda_values * violation, torch.zeros_like(violation), reduction="mean")
        projection_loss = F.mse_loss(mean_logits, proj_mean_logits,reduction="mean")

        # Total loss
        total_loss = (
                surrogate_loss
                + self.ppo_cfg["vf_lambda"] * value_loss
                - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                + self.ppo_cfg["feasibility_lambda"] * feasibility_loss
                + self.ppo_cfg["projection_lambda"] * projection_loss
        ).mean()

        # Collect metrics for logging
        metrics = {
            # loss logging
            "loss": total_loss,
            "surrogate_loss": surrogate_loss,
            "value_loss": self.ppo_cfg["vf_lambda"] * value_loss,
            "entropy": -self.ppo_cfg["entropy_lambda"] * entropy.mean(),
            "feasibility_loss": self.ppo_cfg["feasibility_lambda"] * feasibility_loss,
            "projection_loss": self.ppo_cfg["projection_lambda"] * projection_loss,
            "return": returns.mean(),
            "ratios": ratios.mean(),
            "clipped_ratios": clipped_ratios.mean(),
            "adv": adv,
            "value_pred": value_preds,
            "violations": violation.mean(dim=0).sum(),  # total violation during n-steps
            # performance metrics
            "total_loaded": td["state"]["total_loaded"].mean(),
            "total_profit":  td["state"]["total_revenue"].mean() - td["state"]["total_costs"].mean(),
            "total_revenue": td["state"]["total_revenue"].mean(),
            "total_costs": td["state"]["total_costs"].mean(),
        }
        return total_loss, metrics

    def _aggregate_metrics(self, list_metrics):
        """Aggregate metrics across PPO inner epochs and mini-batches."""
        return {k: torch.stack([dic[k] for dic in list_metrics], dim=0) for k in list_metrics[0]}

def check_for_nans(tensor, name):
    """Check for NaNs and Infs in a tensor and raise an error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")

def check_tensors_for_nans(td, parent_key=""):
    """Recursive check for NaNs and Infs in e.g. TensorDicts and raise an error if found."""
    for key, value in td.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, torch.Tensor):
            check_for_nans(value, full_key)
        elif isinstance(value, TensorDict):
            # Recursively check nested TensorDicts
            check_tensors_for_nans(value, full_key)

def initialize_stats_from_rollouts(env, normalizer, batch_size, num_rollouts=100, device="cuda"):
    """
    Perform random rollouts to initialize mean and std in ObservationNormalizer.

    Args:
        env: The environment instance.
        normalizer: The ObservationNormalizer instance.
        rollout_mpp: Function that performs a rollout and returns observation TensorDict.
        num_rollouts: Number of rollouts to perform.
        device: Device for computations (e.g., "cuda").
    """

    # Test the environment
    def random_action_policy(td):
        """Helper function to select a random action from available actions"""
        batch_size = td.batch_size
        action = torch.distributions.Uniform(env.action_spec.low, env.action_spec.high).sample(batch_size)
        td.set("action", action.to(torch.float16))
        return td

    # Collect observations across rollouts
    collected_obs = {key: [] for key in normalizer.mean.keys()}
    for idx in range(num_rollouts):
        td = env.reset(batch_size=batch_size,)
        while not td["done"].all():
            td = random_action_policy(td)
            td = env.step(td)["next"]
            # Collect each observation field in the TensorDict
            for key, value in td["obs"].items():
                collected_obs[key].append(value)

    # Calculate and set initial mean and std in the normalizer
    for key, values in collected_obs.items():
        # Stack values along a new dimension and calculate mean/std
        values_stack = torch.stack(values, dim=0)  # Shape: [num_rollouts, *obs_shape]
        mean = values_stack.mean(dim=0)
        std = values_stack.std(dim=0)

        # Set initial mean and std in the normalizer with clamping on std
        normalizer.mean[key].data = mean.to(device)
        normalizer.std[key].data = torch.clamp(std, min=1e-6).to(device)  # Avoid near-zero std
