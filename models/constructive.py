import abc

from typing import Any, Callable, Optional, Tuple, Union
import torch
import torch.nn as nn

from tensordict import TensorDict
from torch import Tensor

from rl4co.envs import RL4COEnvBase, get_env
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

from models.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
    calculate_gaussian_entropy,
)
from environment.utils import compute_violation
from models.projection import ProjectionFactory

log = get_pylogger(__name__)


class ConstructiveEncoder(nn.Module, metaclass=abc.ABCMeta):
    """Base class for the encoder of constructive models"""

    @abc.abstractmethod
    def forward(self, td: TensorDict) -> Tuple[Any, Tensor]:
        """Forward pass for the encoder

        Args:
            td: TensorDict containing the input data

        Returns:
            Tuple containing:
              - latent representation (any type)
              - initial embeddings (from feature space to embedding space)
        """
        raise NotImplementedError("Implement me in subclass!")


class ConstructiveDecoder(nn.Module, metaclass=abc.ABCMeta):
    """Base decoder model for constructive models. The decoder is responsible for generating the logits for the action"""

    @abc.abstractmethod
    def forward(
        self, td: TensorDict, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[Tensor, Tensor]:
        """Obtain logits for current action to the next ones

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder. Can be any type
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the logits and the action mask
        """
        raise NotImplementedError("Implement me in subclass!")

    def pre_decoder_hook(
        self, td: TensorDict, env: RL4COEnvBase, hidden: Any = None, num_starts: int = 0
    ) -> Tuple[TensorDict, Any, RL4COEnvBase]:
        """By default, we don't need to do anything here.

        Args:
            td: TensorDict containing the input data
            hidden: Hidden state from the encoder
            env: Environment for decoding
            num_starts: Number of starts for multistart decoding

        Returns:
            Tuple containing the updated hidden state, TensorDict, and environment
        """
        return td, env, hidden


class NoEncoder(ConstructiveEncoder):
    """Default encoder decoder-only models, i.e. autoregressive models that re-encode all the state at each decoding step."""

    def forward(self, td: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return Nones for the hidden state and initial embeddings"""
        return None, None


class ConstructivePolicyMPP(nn.Module):
    """
    Base class for constructive policies. Constructive policies take as input and instance and output a solution (sequence of actions).
    "Constructive" means that a solution is created from scratch by the model.

    The structure follows roughly the following steps:
        1. Create a hidden state from the encoder
        2. Initialize decoding strategy (such as greedy, sampling, etc.)
        3. Decode the action given the hidden state and the environment state at the current step
        4. Update the environment state with the action. Repeat 3-4 until all sequences are done
        5. Obtain log likelihood, rewards etc.

    Note that an encoder is not strictly needed (see :class:`NoEncoder`).). A decoder however is always needed either in the form of a
    network or a function.

    Note:
        There are major differences between this decoding and most RL problems. The most important one is
        that reward may not defined for partial solutions, hence we have to wait for the environment to reach a terminal
        state before we can compute the reward with `env.get_reward()`.

    Warning:
        We suppose environments in the `done` state are still available for sampling. This is because in NCO we need to
        wait for all the environments to reach a terminal state before we can stop the decoding process. This is in
        contrast with the TorchRL framework (at the moment) where the `env.rollout` function automatically resets.
        You may follow tighter integration with TorchRL here: https://github.com/ai4co/rl4co/issues/72.

    Args:
        encoder: Encoder to use
        decoder: Decoder to use
        env_name: Environment name to solve (used for automatically instantiating networks)
        temperature: Temperature for the softmax during decoding
        tanh_clipping: Clipping value for the tanh activation (see Bello et al. 2016) during decoding
        mask_logits: Whether to mask the logits or not during decoding
        train_decode_type: Decoding strategy for training
        val_decode_type: Decoding strategy for validation
        test_decode_type: Decoding strategy for testing
    """

    def __init__(
        self,
        encoder: Union[ConstructiveEncoder, Callable],
        decoder: Union[ConstructiveDecoder, Callable],
        env_name: str = "tsp",
        temperature: float = 1.0,
        tanh_clipping: float = 0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        **unused_kw,
    ):
        super(ConstructivePolicyMPP, self).__init__()

        if len(unused_kw) > 0:
            log.error(f"Found {len(unused_kw)} unused kwargs: {unused_kw}")

        self.env_name = env_name

        # Encoder and decoder
        if encoder is None:
            log.warning("`None` was provided as encoder. Using `NoEncoder`.")
            encoder = NoEncoder()
        self.encoder = encoder
        self.decoder = decoder

        # Decoding strategies
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.tanh_squashing = unused_kw.get("tanh_squashing", False)
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def forward(
        self,
        td: TensorDict,
        env: Optional[Union[str, RL4COEnvBase]] = None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = False,
        return_entropy: bool = False,
        return_feasibility: bool = True,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        """Forward pass of the policy.

        Args:
            td: TensorDict containing the environment state
            env: Environment to use for decoding. If None, the environment is instantiated from `env_name`. Note that
                it is more efficient to pass an already instantiated environment each time for fine-grained control
            phase: Phase of the algorithm (train, val, test)
            calc_reward: Whether to calculate the reward
            return_actions: Whether to return the actions
            return_entropy: Whether to return the entropy
            return_hidden: Whether to return the hidden state
            return_init_embeds: Whether to return the initial embeddings
            return_sum_log_likelihood: Whether to return the sum of the log likelihood
            actions: Actions to use for evaluating the policy.
                If passed, use these actions instead of sampling from the policy to calculate log likelihood
            max_steps: Maximum number of decoding steps for sanity check to avoid infinite loops if envs are buggy (i.e. do not reach `done`)
            decoding_kwargs: Keyword arguments for the decoding strategy. See :class:`rl4co.utils.decoding.DecodingStrategy` for more information.

        Returns:
            out: Dictionary containing the reward, log likelihood, and optionally the actions and entropy
        """

        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Instantiate environment if needed
        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        # Get decode type depending on phase and whether actions are passed for evaluation
        decode_type = decoding_kwargs.pop("decode_type", None)
        # todo: improve here for all continuous
        if actions is not None and env.name == "mpp":
            decode_type = "continuous_evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        project_per_port = decoding_kwargs.get("projection_kwargs", {}).get("project_per_port", False)
        if project_per_port:
            # todo: necessary to prevent double projection
            decode_type = "continuous_sampling"

        # Setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            tanh_squashing=decoding_kwargs.pop("tanh_squashing", self.tanh_squashing),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            env=env,
            **decoding_kwargs,
        )
        decode_strategy_: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            tanh_squashing=decoding_kwargs.pop("tanh_squashing", self.tanh_squashing),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            env=env,
            **decoding_kwargs,
        )
        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        if project_per_port:
            # Perform portwise decoding to construct the solution
            self._portwise_decoding(td, env, actions=actions, hidden=hidden, num_starts=num_starts,
                                    decode_strategy=decode_strategy, decode_strategy_=decode_strategy_,max_steps=max_steps,
                                    **decoding_kwargs)
        else:
            # Perform main decoding to construct the solution
            self._main_decoding(td, env, actions=actions, hidden=hidden, num_starts=num_starts,
                                decode_strategy=decode_strategy, max_steps=max_steps,)

        # Post-decoding hook: used for the final step(s) of the decoding strategy
        dictout, td, env = decode_strategy.post_decoder_hook(td, env)
        logprobs, actions = dictout["logprobs"], dictout["actions"]
        # Output dictionary construction
        if calc_reward:
            if env.name == "mpp":
                td.set("reward", env.get_reward(td, dictout["utilization"]))
            else:
                td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                dictout["logprobs"], dictout["actions"],
                mask=dictout["action_masks"],
                return_sum=return_sum_log_likelihood),
            "utilization": dictout["utilization"],
            "total_loaded": td["state"].get("total_loaded", None),
            "total_revenue": td["state"].get("total_revenue", None),
            "total_cost": td["state"].get("total_cost", None),
        }

        if return_actions:
            outdict["actions"] = actions
            outdict["action_masks"] = dictout["action_masks"]
        if return_entropy:
            if decode_type.startswith("continuous"):
                outdict["entropy"] = calculate_gaussian_entropy((dictout["proj_mean_logits"], dictout["std_logits"]))
            else:
                outdict["entropy"] = calculate_entropy(logprobs)
        if return_feasibility:
            outdict["lhs_A"], outdict["rhs"] = dictout["lhs_A"], dictout["rhs"]
            outdict["violation"] = compute_violation(dictout["actions"].unsqueeze(-2), dictout["lhs_A"], dictout["rhs"])
            outdict["reward"] -= outdict["violation"].sum(dim=(-1, -2)).view(td.batch_size[0], 1)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds

        return outdict

    def _main_decoding(self, td: TensorDict, env: RL4COEnvBase, actions=None, hidden: Any = None, num_starts: int = 0,
                       decode_strategy: DecodingStrategy = None, max_steps=1_000_000, use_lp_solver=False,
                       **decoding_kwargs):
        # Main decoding: loop until all sequences are done
        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[:, step, ] if actions is not None else None,
            )
            td = env.step(td)["next"]

            step += 1
            if step > max_steps:
                log.error(
                    f"Exceeded maximum number of steps ({max_steps}) during decoding"
                )
                break

    def _portwise_decoding(self, td: TensorDict, env: RL4COEnvBase, actions=None, hidden: Any = None, num_starts: int = 0,
                           decode_strategy: DecodingStrategy = None, decode_strategy_: DecodingStrategy = None,
                           max_steps=1_000_000, use_lp_solver=False, **decoding_kwargs):
        # Setup projection layer
        batch_size = td.batch_size[0]
        projection_layer = ProjectionFactory.create_class(decoding_kwargs.get("projection_type", {}),
                                                          kwargs=decoding_kwargs.get("projection_kwargs", {}))
        # Perform decoding per port to construct the solution
        step = 0
        # Create constraint matrix for portwise decoding
        self._constraint_shapes_portwise(env)
        port_A = self._create_port_constraint_matrix(td, env.stability_params_lhs)
        port_actions = torch.zeros(batch_size, env.P-1, self.features_class_pod, self.features_action, device=td.device)
        port_rhs = torch.zeros(batch_size, env.P-1, self.n_constraints, device=td.device)
        for port in range(env.P - 1):
            steps_port = (env.P - (port + 1)) * env.K

            ## Decode for each port based on td
            port_td = td.copy()  # Use copy to avoid changing the original td
            for t in range(steps_port):
                # Predict actions for the port
                logits, mask = self.decoder(port_td, hidden, num_starts)
                port_td = decode_strategy_.step(logits, mask, port_td, action=None)
                # Store actions and rhs in portwise tensors
                port_actions[:, port, t] = port_td["action"].clone()
                port_rhs[:, port, t] = port_td["rhs"][...,0]
                # Only store stability rhs at start port
                if t == 0: port_rhs[:, port, -4:] = port_td["rhs"][...,1:]
                # Perform step
                port_td = env.step(port_td)["next"]

            ## Project actions to feasible region
            # todo: demand violation works, stability violation does not work
            #  - check if stability is correctly computed
            proj_actions = projection_layer(port_actions[:, port].view(-1, self.features_action*self.features_class_pod),
                                            port_A[:, port], port_rhs[:, port],
                                            alpha=1e-4, delta=1e-2, tolerance=1e-2, max_iter=1000,)
            proj_actions = proj_actions.view(batch_size, self.features_class_pod, self.features_action)

            ## Decode with projected actions
            for t in range(steps_port):
                logits, mask = self.decoder(td, hidden, num_starts)
                td = decode_strategy.step(logits, mask, td, action=proj_actions[:, t])
                td = env.step(td)["next"]
                step += 1

    def _constraint_shapes_portwise(self, env: RL4COEnvBase):
        self.steps = env.P - 1
        self.pods = env.P - 1
        self.n_constraints = env.K * (env.P - 1) + env.n_constraints - 1
        self.features_action = env.B * env.D
        self.features_class_pod = env.K * (env.P - 1)

    def _create_port_constraint_matrix(self, td: TensorDict, stability_params:Tensor):
        """
        Create the constraint matrix for the portwise decoding:
        - A: constraint matrix in shape         [P-1, n_constraints, features ]
        - b: rhs of the constraints in shape    [Batch, P-1, n_constraints]
        - x: variable in shape                  [Batch, P-1, features]
        Features = B * D * K * (P-1)
        - Inner loop of features is all features of action, outer loop is features of class and pod
        - First, k=0 for (b*d) features, then k=1 for next (b*d) features, etc.
        """
        batch_size = td.batch_size[0]
        stab_params = stability_params.permute(0,2,1).view(1, 1, 4, -1, self.features_action).repeat(1, 1, 1, self.pods, 1)
        A = torch.zeros((batch_size, self.steps, self.n_constraints, self.features_class_pod, self.features_action, ),
                        device=td.device, dtype=torch.float32)
        for i in range(self.n_constraints):
            if i < self.features_class_pod:
                A[:, :, i, i, :] = 1
        A[:, :, -4:, :, :] = stab_params
        output = A.view(batch_size, self.steps, self.n_constraints, self.features_action*self.features_class_pod)
        return output

def stack_dicts_on_dim(dicts, dim=1):
    """Stack a list of dictionaries on the first dimension of the tensors"""
    stacked_dict = {}
    for key in dicts[0].keys():
        tensors_to_stack = [d[key] for d in dicts]
        stacked_tensor = torch.cat(tensors_to_stack, dim=dim)
        stacked_dict[key] = stacked_tensor
    return stacked_dict