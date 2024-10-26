from typing import Callable, Dict
from tensordict import TensorDict

import torch
import torch.nn as nn

from rl4co.models.common.constructive.autoregressive.policy import AutoregressivePolicy
from rl4co.models.zoo.am.decoder import AttentionModelDecoder
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from rl4co.models.zoo.am.policy import AttentionModelPolicy

# Custom modules
from models.decoding import DecodingStrategy, get_decoding_strategy, get_log_likelihood, calculate_gaussian_entropy

class AttentionModelPolicy4PPO(AttentionModelPolicy):
    def __init__(
        self,
        encoder: nn.Module = None,
        decoder: nn.Module = None,
        embed_dim: int = 128,
        num_encoder_layers: int = 3,
        num_heads: int = 8,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        env_name: str = "tsp",
        encoder_network: nn.Module = None,
        init_embedding: nn.Module = None,
        context_embedding: nn.Module = None,
        dynamic_embedding: nn.Module = None,
        use_graph_context: bool = True,
        linear_bias_decoder: bool = False,
        sdpa_fn: Callable = None,
        sdpa_fn_encoder: Callable = None,
        sdpa_fn_decoder: Callable = None,
        mask_inner: bool = True,
        out_bias_pointer_attn: bool = False,
        check_nan: bool = True,
        temperature: float = 1.0,
        tanh_clipping: float = 10.0,
        mask_logits: bool = True,
        train_decode_type: str = "sampling",
        val_decode_type: str = "greedy",
        test_decode_type: str = "greedy",
        moe_kwargs: dict = {"encoder": None, "decoder": None},
        **unused_kwargs,
    ):
        # Pass everything necessary to the parent class
        super(AttentionModelPolicy4PPO, self).__init__(
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            env_name=env_name,
            encoder_network=encoder_network,
            init_embedding=init_embedding,
            context_embedding=context_embedding,
            dynamic_embedding=dynamic_embedding,
            use_graph_context=use_graph_context,
            linear_bias_decoder=linear_bias_decoder,
            sdpa_fn=sdpa_fn,
            sdpa_fn_encoder=sdpa_fn_encoder,
            sdpa_fn_decoder=sdpa_fn_decoder,
            mask_inner=mask_inner,
            out_bias_pointer_attn=out_bias_pointer_attn,
            check_nan=check_nan,
            temperature=temperature,
            tanh_clipping=tanh_clipping,
            mask_logits=mask_logits,
            train_decode_type=train_decode_type,
            val_decode_type=val_decode_type,
            test_decode_type=test_decode_type,
            moe_kwargs=moe_kwargs,
            **unused_kwargs,
        )
        # Projection layer
        self.projection_type = unused_kwargs.pop("projection_type", None) # pop before passing to super
        self.projection_kwargs = unused_kwargs.pop("projection_kwargs", None) # pop before passing to super
        # Decode kwargs
        self.temperature = temperature
        self.tanh_clipping = tanh_clipping
        self.mask_logits = mask_logits
        self.train_decode_type = train_decode_type
        self.val_decode_type = val_decode_type
        self.test_decode_type = test_decode_type

    def act(self, td, env, phase: str = "train", action=None, **kwargs) -> TensorDict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Get decode type depending on phase and whether actions are passed for evaluation
        if action is not None:
            decode_type = "continuous_evaluate"
        else:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=self.temperature,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
            projection_type = self.projection_type,
            projection_kwargs = self.projection_kwargs,
        )
        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main step decoding
        logits, mask = self.decoder(td, hidden, num_starts)
        td = decode_strategy.step(
            logits,
            mask,
            td,
            action=action if action is not None else None,
            scale_factor=td["scale_factor"],
        )
        return td

    def evaluate(self, td, env, phase: str = "train", action=None, **kwargs) -> TensorDict:
        # Encoder: get encoder output and initial embeddings from initial state
        hidden, init_embeds = self.encoder(td)

        # Get decode type depending on phase and whether actions are passed for evaluation
        if action is not None:
            decode_type = "continuous_evaluate"
        else:
            decode_type = getattr(self, f"{phase}_decode_type")

        # Setup decoding strategy
        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=self.temperature,
            tanh_clipping=self.tanh_clipping,
            mask_logits=self.mask_logits,
            projection_type = self.projection_type,
            projection_kwargs = self.projection_kwargs,
        )
        # Pre-decoding hook: used for the initial step(s) of the decoding strategy
        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)
        # Additionally call a decoder hook if needed before main decoding
        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        # Main step decoding
        logits, mask = self.decoder(td, hidden, num_starts)
        td = decode_strategy.step(
            logits,
            mask,
            td,
            action=action if action is not None else None,
            scale_factor=td["scale_factor"],
        )
        # Compute entropy
        td["entropy"] = calculate_gaussian_entropy((td["proj_mean_logits"], td["std_logits"]))
        return td

    @torch.no_grad()
    def generate(self, td, env=None, phase: str = "train", **kwargs) -> dict:
        assert phase != "train", "dont use generate() in training mode"
        with torch.no_grad():
            out = super().__call__(td, env, phase=phase, **kwargs)
        return out