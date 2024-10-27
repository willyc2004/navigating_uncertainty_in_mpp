from typing import Any, Union
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchrl.objectives.value.functional import generalized_advantage_estimate

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

class n_step_Memory:
    def __init__(self, batch_size, n_step, env, device="cuda"):
        self.ll = torch.zeros((*batch_size, n_step, env.D * env.B), device=device)
        self.value_preds = torch.zeros((*batch_size, n_step, 1), device=device)
        self.entropy = torch.zeros((*batch_size, n_step, 1), device=device)
        self.mean_logits = torch.zeros((*batch_size, n_step, env.D * env.B), device=device)
        self.std_logits = torch.zeros((*batch_size, n_step, env.D * env.B), device=device)
        self.proj_mean_logits = torch.zeros((*batch_size, n_step, env.D * env.B), device=device)
        self.lhs_A = torch.zeros((*batch_size, n_step, env.n_constraints, env.D * env.B), device=device)
        self.rhs = torch.zeros((*batch_size, n_step, env.n_constraints), device=device)
        self.dones = torch.zeros((*batch_size, n_step, 1), device=device)

    def clear_memory(self):
        self.ll = self.ll.new_zeros(self.ll.size())
        self.value_preds = self.value_preds.new_zeros(self.value_preds.size())
        self.entropy = self.entropy.new_zeros(self.entropy.size())
        self.mean_logits = self.mean_logits.new_zeros(self.mean_logits.size())
        self.std_logits = self.std_logits.new_zeros(self.std_logits.size())
        self.proj_mean_logits = self.proj_mean_logits.new_zeros(self.proj_mean_logits.size())
        self.lhs_A = self.lhs_A.new_zeros(self.lhs_A.size())
        self.rhs = self.rhs.new_zeros(self.rhs.size())
        self.dones = self.dones.new_zeros(self.dones.size())

class Projection_Nstep_PPO(RL4COLitModule):
    """
    An implementation of the n-step dactProximal Policy Optimization (PPO) algorithm (https://arxiv.org/abs/2110.02544)
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
        projection_lambda: float = 1.0,  # lambda of projection loss
        normalize_adv: bool = False,  # whether to normalize advantage
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
        n_step: float = 36,  # n-step for n-step PPO
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
        self.projection_layer = kwargs.pop("projection_type", None)  # pop before passing to super
        self.projection_kwargs = kwargs.pop("projection_kwargs", None)  # pop before passing to super
        super().__init__(
            env,
            policy,
            metrics=metrics,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            lr_scheduler_interval=lr_scheduler_interval,
            lr_scheduler_monitor=lr_scheduler_monitor,
            optimizer_kwargs={"lr": lr},
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
            "max_grad_norm": max_grad_norm,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "n_step": n_step,
            "T_train": T_train,
            "T_test": T_test,
            "lr": lr,
            "entropy_lambda": entropy_lambda,
            "feasibility_lambda": feasibility_lambda,
            "projection_lambda": projection_lambda,
            "update_timestep": update_timestep,
            "kl_threshold": kl_threshold,
            "kl_penalty_lambda": kl_penalty_lambda,
        }
        self.lambda_violations = torch.tensor([demand_lambda] + [stability_lambda] * env.n_stability,
                                              device='cuda', dtype=torch.float32)

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

    def shared_step(
        self, batch: Any, batch_idx: int, phase: str, dataloader_idx: int = None
    ):
        if phase != "train":
            td = self.env.reset(batch)
            out = self.policy.generate(td, env=self.env, phase=phase, return_feasibility=True)
        else:
            # init the training
            memory = Memory(batch.batch_size, self.ppo_cfg["n_step"],self.env)
            n_step_memory = n_step_Memory(batch.batch_size, self.ppo_cfg["n_step"], self.env)
            td = self.env.reset(batch)
            out = self.update(td, memory, n_step_memory, phase, device="cuda")
        metrics = self.log_metrics(out, phase, dataloader_idx=dataloader_idx)
        return {"loss": out.get("loss", None), **metrics}

    def update(self, td, memory, step_memory, phase, device):
        # perform gradiant updates every n_step until reaching T_max
        assert (
                self.ppo_cfg["T_train"] % self.ppo_cfg["n_step"] == 0
        ), "T_max should be divided by n_step with no remainder"
        t = 0
        batch_size = td.batch_size
        agg_reward = torch.zeros((*batch_size, 1), device=device)
        list_metrics = []
        while t < self.ppo_cfg["T_train"]:
            memory.clear_memory()

            # Rollout for n_step; store the information in memory
            with torch.no_grad():
                for i in range(self.ppo_cfg["n_step"]):
                    memory.tds.append(td.clone())
                    td = self.policy.act(memory.tds[i], self.env, phase=phase,)
                    memory.values[:,i] = self.critic(memory.tds[i]).view(-1, 1)
                    td = self.env.step(td.clone())["next"] # must clone to avoid in-place operation
                    memory.actions[:,i] = td["action"]
                    memory.logprobs[:,i] = td["logprobs"]
                    memory.rewards[:,i] = td["reward"].view(-1, 1)
                    memory.profit[:,i] = td["profit"].view(-1, 1)

            t += self.ppo_cfg["n_step"]

            # PPO inner epoch, K
            for k in range(self.ppo_cfg["ppo_epochs"]):
                # with torch.autograd.detect_anomaly():
                step_memory.clear_memory()
                for i in range(self.ppo_cfg["n_step"]):
                    out = self.policy.evaluate(
                        memory.tds[i],
                        action=memory.actions[:,i],
                        env=self.env,
                        phase=phase,
                        return_actions=False,
                    )
                    # Store all metrics of new policy
                    step_memory.value_preds[:,i] = self.critic(memory.tds[i]).view(-1, 1)
                    step_memory.dones[:,i] = out["done"].view(-1,1)
                    step_memory.ll[:,i] = out["logprobs"]
                    step_memory.entropy[:,i] = out["entropy"].view(-1, 1)
                    step_memory.mean_logits[:,i] = out["mean_logits"]
                    step_memory.std_logits[:,i] = out["std_logits"]
                    step_memory.proj_mean_logits[:,i] = out["proj_mean_logits"]
                    step_memory.lhs_A[:,i] = out["lhs_A"]
                    step_memory.rhs[:,i] = out["rhs"]

                # ratio is always 1 for first ppo epoch
                if k == 0:
                    step_memory.ll = memory.logprobs  # [batch, n_step, D*B]
                    old_ll = memory.logprobs # [batch, n_step, D*B]
                    old_values = memory.values  # [batch, n_step, 1]
                    rewards = memory.rewards  # [batch, n_step, 1]
                    agg_reward += rewards.sum(dim=1)  # [batch, 1]

                # prepare metrics for computation
                ll = step_memory.ll
                value_preds = step_memory.value_preds
                entropy = step_memory.entropy
                mean_logits = step_memory.mean_logits
                proj_mean_logits = step_memory.proj_mean_logits
                # std_logits = step_memory.std_logits
                lhs_A = step_memory.lhs_A
                rhs = step_memory.rhs
                dones = step_memory.dones

                # Compute the sample importance ratio of new and old actions
                log_ratios = (ll - old_ll.detach()) # [batch, n_step, F]
                ratios = torch.exp(log_ratios) # [batch, n_step, F]
                clipped_ratios = torch.clamp(
                    ratios,
                    1 - self.ppo_cfg["clip_range"],
                    1 + self.ppo_cfg["clip_range"],
                )

                # Compute GAE
                adv, returns = generalized_advantage_estimate(
                    gamma=self.ppo_cfg["gamma"], lmbda=self.ppo_cfg["gae_lambda"],state_value=old_values,
                    next_state_value=value_preds, reward=rewards, done=dones.bool(),)

                # Normalize advantage
                if self.ppo_cfg["normalize_adv"]:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # Compute the surrogate loss
                surrogate_loss = -torch.min(ratios * adv, clipped_ratios * adv,).mean()

                # compute value and entropy loss
                value_loss = F.huber_loss(value_preds.detach(), returns, reduction='mean')

                # Custom loss terms are computed with the mean logit, as it is part of the computation graph.
                # Note that actions are sampled from policy distribution derived from logits, i.e. detached from
                # the computation graph. Instead, we use the mean to compute distances;
                # - Feasibility loss computes distance between the feasible set and the project policy distribution
                # - Projection loss computes the distance between the raw and projected policy distributions
                lhs = (lhs_A * proj_mean_logits.unsqueeze(-2)).sum(dim=-1)
                violation = torch.clamp(lhs - rhs, min=0)
                feasibility_loss = F.mse_loss(self.lambda_violations * violation, torch.zeros_like(violation),
                                              reduction="mean")
                projection_loss = F.mse_loss(mean_logits, proj_mean_logits, reduction="mean")

                # compute total loss:
                #   max surrogate loss, entropy,
                #   min value loss, feasibility, and projection loss
                loss = (
                        surrogate_loss
                        + self.ppo_cfg["vf_lambda"] * value_loss
                        - self.ppo_cfg["entropy_lambda"] * entropy.mean() # entropy loss works for surrogate loss
                        + self.ppo_cfg["feasibility_lambda"] * feasibility_loss
                        + self.ppo_cfg["projection_lambda"] * projection_loss
                ).mean()

            # perform manual optimization following the Lightning routine
            # https://lightning.ai/docs/pytorch/stable/common/optimization.html
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

            # Return metrics of last ppo epoch
            list_metrics.append({
                    # main logging
                    "loss": loss,
                    "surrogate_loss": surrogate_loss,
                    "value_loss": self.ppo_cfg["vf_lambda"] * value_loss,
                    "entropy": -self.ppo_cfg["entropy_lambda"] * entropy,
                    "feasibility_loss": self.ppo_cfg["feasibility_lambda"] * feasibility_loss,
                    "projection_loss": self.ppo_cfg["projection_lambda"] * projection_loss,
                    # extra logging
                    "episodic_reward": agg_reward.mean(), # mean over batch and n_step
                    "return": returns.mean(),
                    "ratios": ratios.mean(),
                    "clipped_ratios": clipped_ratios.mean(),
                    # "kl_div": kl_divergence.mean(),
                    "adv": adv,
                    "value_pred": value_preds,
                    "total_loaded": td["state"]["total_loaded"].mean(),
                    "violations": violation.mean(dim=0).sum(), # total violation
                    # "ll": ll.sum(),
                    # "old_ll": old_ll.sum(),
                    # "action": action.mean(),
                    # profit metrics
                    # "revenue": revenue.mean(),
                    # "costs": costs.mean(),
                })
        dict_metrics = {k: torch.stack([dic[k] for dic in list_metrics], dim=0) for k in list_metrics[0]}
        return dict_metrics

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")