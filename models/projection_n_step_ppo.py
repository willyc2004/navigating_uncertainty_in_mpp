from typing import Any, Union
import copy
from tensordict import TensorDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
from torchrl.objectives.value.functional import generalized_advantage_estimate
from torch.utils.data import BatchSampler, SubsetRandomSampler

# Enable CUDA_LAUNCH_BLOCKING for debugging CUDA errors
import os
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
from rl4co.utils.ops import gather_by_index
log = get_pylogger(__name__)

# Custom imports
from environment.utils import compute_violation

class Memory:
    def __init__(self, td, max_steps, env, device="cuda"):
        batch_size = td.batch_size
        self.tds_obs = []
        self.actions = torch.zeros((*batch_size, max_steps, env.D * env.B), device=device)
        self.logprobs = torch.zeros((*batch_size, max_steps, env.D * env.B), device=device)
        self.rewards = torch.zeros((*batch_size, max_steps, 1), device=device)
        self.values = torch.zeros((*batch_size, max_steps, 1), device=device)
        self.profit = torch.zeros((*batch_size, max_steps, 1), device=device)

    def clear_memory(self):
        del self.tds_obs[:]
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

            # Update running statistics with momentum for stability
            self.mean[key].data = self.momentum * batch_mean + (1 - self.momentum) * self.mean[key].data
            self.std[key].data = self.momentum * batch_std + (1 - self.momentum) * self.std[key].data

class BatchDataset(Dataset):
    def __init__(self, td, memory):
        """
        Custom dataset for PPO mini-batch sampling.
        Args:
            td (TensorDict): Training data containing the current state, actions, etc.
            memory (object): Memory object holding tensors for actions, logprobs, rewards, etc.
        """
        self.td = td
        self.memory = memory

    def __len__(self):
        """Return the batch size (number of samples)."""
        return self.td.shape[0]

    def __getitem__(self, index):
        """Return data for a single index."""
        stacked_obs = torch.stack(self.memory.tds_obs, dim=0).permute(1,0)
        return {
            "actions": self.memory.actions[index],
            "values": self.memory.values[index],
            "logprobs": self.memory.logprobs[index],
            "rewards": self.memory.rewards[index],
            "profit": self.memory.profit[index],
            "tds": stacked_obs[index],
        }

def recursive_tensordict_collate(batch):
    """
    Recursively collates a batch of TensorDicts, including nested TensorDicts.
    """
    if isinstance(batch[0], TensorDict):
        # Handle nested TensorDicts
        keys = batch[0].keys()
        collated_data = {}
        for key in keys:
            # Collect the data for this key across all TensorDicts
            key_batch = [td[key] for td in batch]
            # Recursively collate the data for this key
            collated_data[key] = recursive_tensordict_collate(key_batch)
        # Return a stacked TensorDict
        return TensorDict(collated_data, batch_size=[len(batch)])

    elif isinstance(batch[0], (list, tuple)):
        # If the items are lists or tuples, collate each element recursively
        return type(batch[0])(recursive_tensordict_collate(items) for items in zip(*batch))

    elif isinstance(batch[0], dict):
        # If the items are dictionaries, collate each key recursively
        return {key: recursive_tensordict_collate([d[key] for d in batch]) for key in batch[0]}

    else:
        # For standard tensors or values, use default stacking (e.g., for torch.Tensor)
        return default_collate(batch)

def tensordict_collate_fn(batch):
    """
    Collate function for a batch of TensorDicts, including support for nested TensorDicts.
    """
    if not isinstance(batch, list):
        raise TypeError("Batch should be a list of TensorDicts.")
    return recursive_tensordict_collate(batch)


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
            "train": [
                # loss logging
                "loss", "surrogate_loss", "value_loss", "entropy", "feasibility_loss", "projection_loss",
                "return", "ratios", "clipped_ratios", "adv", "value_pred",
                # violation logging
                "mean_violation", "violation",
                "violation_demand", "violation_lcg_ub", "violation_lcg_lb", "violation_vcg_ub", "violation_vcg_lb",
                # performance metrics
                "total_loaded", "total_profit", "total_revenue", "total_cost",
            ]
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
        self.decoder_type = kwargs.pop("decoder_type", None)  # pop before passing to super
        self.env_kwargs = kwargs.pop("env_kwargs", None)  # pop before passing to super
        self.model_kwargs = kwargs.pop("model_kwargs", None)  # pop before passing to super
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
        self.select_obs_td = ["obs", "done", "timestep", "action_mask", "lhs_A", "rhs", "clip_min", "clip_max",
                              ("state", "utilization")]

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
            # Log metrics
            # self.log("val/reward", out["reward"].mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/total_revenue", out["total_revenue"].mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/total_cost", out["total_cost"].mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/total_loaded", out["total_loaded"].mean(), on_epoch=True, prog_bar=True, logger=True)

            # Log relevant violations
            relevant_violation = self._get_relevant_violations(out["violation"], self.env)
            self.log("val/violation_demand", relevant_violation[...,0].sum(dim=1).mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/violation_lcg_ub", relevant_violation[...,1].sum(dim=1).mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/violation_lcg_lb", relevant_violation[...,2].sum(dim=1).mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/violation_vcg_ub", relevant_violation[...,3].sum(dim=1).mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/violation_vcg_lb", relevant_violation[...,4].sum(dim=1).mean(), on_epoch=True, prog_bar=True, logger=True)
            self.log("val/violation", relevant_violation.sum(dim=(1,2)).mean(), on_epoch=True, prog_bar=True, logger=True)
        else:
            td = self.env.reset(batch)
            memory = Memory(batch, self.ppo_cfg["n_step"], self.env)
            out = self.update(td, memory, phase)
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def update(self, td, memory, phase,):
        assert (
                self.ppo_cfg["T_train"] % self.ppo_cfg["n_step"] == 0
        ), "T_train should be divided by n_step with no remainder"
        list_metrics = []

        # Gather n_steps in memory
        memory.clear_memory()
        for i in range(self.ppo_cfg["n_step"]):
            # Store observation in memory
            td_obs = td.select(*self.select_obs_td).clone()
            memory.tds_obs.append(td_obs.clone())

            # Generate actions, perform step in environment and store results
            memory.values[:, i] = self.critic(td_obs).view(-1, 1).detach()
            td = self.policy.act(td.clone(), self.env, phase=phase).detach()
            td = self.env.step(td.clone())["next"]
            memory.actions[:, i] = td["action"].clone()
            memory.logprobs[:, i] = td["logprobs"].clone()
            memory.rewards[:, i] = td["reward"].view(-1, 1).clone()
            memory.profit[:, i] = td["profit"].view(-1, 1).clone()

        # Create DataLoader for mini-batch sampling
        dataset = BatchDataset(td, memory)
        dataloader = DataLoader(dataset, batch_size=self.mini_batch_size, shuffle=True,
                                drop_last=True, collate_fn=tensordict_collate_fn)

        for k in range(self.ppo_cfg["ppo_epochs"]):
            for batch in dataloader:
                # Forward pass based on mini-batch actions
                out = self.policy.act(
                    batch["tds"],
                    action=batch["actions"],
                    env=self.env,
                    phase=phase,
                    return_actions=True,
                )
                out["value_preds"] = self.critic(batch["tds"]).clone()

                # Advantage and return computation
                adv, returns = generalized_advantage_estimate(
                    gamma=self.ppo_cfg["gamma"],
                    lmbda=self.ppo_cfg["gae_lambda"],
                    state_value=batch["values"],
                    next_state_value=out["value_preds"].detach(),
                    reward=batch["rewards"],
                    done=out["done"].bool(),
                )
                out["adv"] = self._normalize_if_enabled(adv, self.ppo_cfg["normalize_adv"])
                out["returns"] = self._normalize_if_enabled(returns, self.ppo_cfg["normalize_return"])
                # print(f"Adv: {out['adv'].mean(dim=0).T}")
                # print(f"Returns: {out['returns'].mean(dim=0).T}")
                # print(f"Values: {batch['values'].mean(dim=0).T}")
                # print(f"Rewards: {batch['rewards'].mean(dim=0).T}")
                # print("#" * 50)
                # print(f"Actions requires_grad: {batch['actions'].requires_grad}")
                # print(f"Old log probs requires_grad: {batch['logprobs'].requires_grad}")
                # print(f"Values requires_grad: {batch['values'].requires_grad}")
                # print(f"Rewards requires_grad: {batch['rewards'].requires_grad}")
                # print(f"Adv requires_grad: {out['adv'].requires_grad}")
                # print(f"Returns requires_grad: {out['returns'].requires_grad}")
                # print("-" * 50)
                # print(f"Log probs requires_grad: {out['logprobs'].requires_grad}")
                # print(f"Value preds requires_grad: {out['value_preds'].requires_grad}")
                # print(f"logits requires_grad: {out['mean_logits'].requires_grad}")
                # print(f"Projected logits requires_grad: {out['proj_mean_logits'].requires_grad}")

                # with torch.autograd.set_detect_anomaly(True):
                # Compute losses
                loss, metrics = self._compute_losses(out, batch, td, self.lambda_violations, )
                # Backward pass
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

    def _compute_losses(self, out, batch, td, lambda_values, tolerance=1e-4, alpha=0.5,):
        """Compute the PPO loss, value loss, entropy loss, feasibility loss, and projection loss."""
        # Extract from memory/out
        old_ll = batch["logprobs"] # Old log probabilities
        value_preds = out["value_preds"]  # Current value predictions
        ll = out["logprobs"]  # Current log probabilities
        entropy = out["entropy"]  # Policy entropy
        mean_logits = out["mean_logits"]  # Current logits
        proj_mean_logits = out["proj_mean_logits"]  # Projected logits
        adv = out["adv"]  # Advantages
        returns = out["returns"]  # Returns

        # Compute the ratios for PPO clipping
        log_ratios = old_ll.detach() - ll  # Detach old log-likelihoods
        ratios = torch.exp(log_ratios.sum(dim=-1, keepdims=True))  # Calculate importance sampling ratios
        clipped_ratios = torch.clamp(ratios, 1 - self.ppo_cfg["clip_range"], 1 + self.ppo_cfg["clip_range"])
        surrogate_loss = -torch.min(ratios * adv, clipped_ratios * adv).mean()  # Surrogate loss

        # Compute the value loss using Huber loss
        value_loss = F.huber_loss(value_preds, returns.detach(), reduction="mean")

        # Compute feasibility and projection losses
        violation = compute_violation(proj_mean_logits.unsqueeze(-2), out["lhs_A"], out["rhs"])  # Feasibility violation
        if self.ppo_cfg["adaptive_feasibility_lambda"]:
            # Adapt lambda values for feasibility constraints
            lambda_values = torch.where(
                violation > tolerance,
                lambda_values * (1 + alpha * violation),
                lambda_values,
            )
        feasibility_loss = F.mse_loss(lambda_values * violation, torch.zeros_like(violation), reduction="mean")  # Feasibility loss
        proj_mean_logits_detached = proj_mean_logits.detach()
        projection_loss = F.mse_loss(mean_logits, proj_mean_logits_detached, reduction="mean")  # Projection loss

        # Combine losses into the total loss
        total_loss = (
                surrogate_loss
                + self.ppo_cfg["vf_lambda"] * value_loss
                - self.ppo_cfg["entropy_lambda"] * entropy.mean()
                + self.ppo_cfg["feasibility_lambda"] * feasibility_loss
                + self.ppo_cfg["projection_lambda"] * projection_loss
        ).mean()
        assert total_loss.requires_grad, "Total loss should require gradients"
        check_for_nans(total_loss, "total_loss")

        # Compute relevant violations for metrics
        real_violation = compute_violation(out["action"].unsqueeze(-2), out["lhs_A"], out["rhs"])
        relevant_violation = self._get_relevant_violations(real_violation.detach(), self.env)
        # Collect metrics for logging
        metrics = {
            # Losses
            "loss": total_loss.detach(),
            "surrogate_loss": surrogate_loss.detach(),
            "value_loss": (self.ppo_cfg["vf_lambda"] * value_loss).detach(),
            "entropy": (-self.ppo_cfg["entropy_lambda"] * entropy.mean()).detach(),
            "feasibility_loss": (self.ppo_cfg["feasibility_lambda"] * feasibility_loss).detach(),
            "projection_loss": (self.ppo_cfg["projection_lambda"] * projection_loss).detach(),
            # Performance metrics
            "return": returns.mean().detach(),
            "ratios": ratios.mean().detach(),
            "clipped_ratios": clipped_ratios.mean().detach(),
            "adv": adv.detach(),
            "value_pred": value_preds.detach(),
            # Violation metrics
            "mean_violation":violation.sum(dim=(1, 2)).mean(),
            "violation": relevant_violation.sum(dim=(1, 2)).mean().detach(),
            "violation_demand": relevant_violation[..., 0].sum(dim=1).mean().detach(),
            "violation_lcg_ub": relevant_violation[..., 1].sum(dim=1).mean().detach(),
            "violation_lcg_lb": relevant_violation[..., 2].sum(dim=1).mean().detach(),
            "violation_vcg_ub": relevant_violation[..., 3].sum(dim=1).mean().detach(),
            "violation_vcg_lb": relevant_violation[..., 4].sum(dim=1).mean().detach(),
            # Additional metrics for debugging or logging
            "total_loaded": td["state"]["total_loaded"].mean().detach(),
            "total_profit": (
                    td["state"]["total_revenue"].mean() - td["state"]["total_cost"].mean()
            ).detach(),
            "total_revenue": td["state"]["total_revenue"].mean().detach(),
            "total_cost": td["state"]["total_cost"].mean().detach(),
        }
        return total_loss, metrics

    def _aggregate_metrics(self, list_metrics):
        """Aggregate metrics across PPO inner epochs and mini-batches."""
        return {k: torch.stack([dic[k] for dic in list_metrics], dim=0) for k in list_metrics[0]}

    def _get_relevant_violations(self, violation, env):
        """Get all demand violations, but only stability at timesteps between ports"""
        relevant_violation = torch.where(env.next_port_mask.view(1, -1, 1) == 1, violation, torch.zeros_like(violation))
        relevant_violation[..., 0] = violation[..., 0]
        return relevant_violation

def recursive_check_for_nans(td, parent_key=""):
    """Recursive check for NaNs and Infs in e.g. TensorDicts and raise an error if found."""
    for key, value in td.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, torch.Tensor):
            check_for_nans(value, full_key)
        elif isinstance(value, TensorDict):
            # Recursively check nested TensorDicts
            recursive_check_for_nans(value, full_key)

def check_for_nans(tensor, name):
    """Check for NaNs and Infs in a tensor and raise an error if found."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")