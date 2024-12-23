import os

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

from typing import Optional
import torch

from environment.models.zoo import AttentionModelPolicy

# RL4CO
from environment.utils.trainer import RL4COTrainer
from environment.models.zoo import AMPPO
# Customized RL4CO modules
from rl4co.ppo import ProjectionPPO
from environment.models.zoo.am.encoder import AttentionModelEncoder
from models.examples.decoder2 import AttentionModelDecoder
from models.rl4co.construct_continuous import ConstructivePolicy
from environment.rl4co.env import MasterPlanningEnv
# AMPPO.__bases__ = (StepwisePPO,)  # Adapt base class
# AMPPO.__bases__ = (Projection_Nstep_PPO,)  # Adapt base class
# AMPPO.__bases__ = (Projection_PPO,)  # Adapt base class
AMPPO.__bases__ = (ProjectionPPO,)  # Adapt base class
AttentionModelPolicy.__bases__ = (ConstructivePolicy,)  # Adapt base class
# AutoregressivePolicy.__bases__ = (ContinuousConstructivePolicy,)
# AutoregressivePolicy.__bases__ = (ConstructivePolicyMPP,)  # Adapt base class
from models.embeddings import MPPContextEmbedding, MPPInitEmbedding

import yaml
from dotmap import DotMap

def adapt_env_kwargs(config):
    """Adapt environment kwargs based on configuration"""
    config.env.bays = 10 if config.env.TEU == 1000 else 20
    config.env.weight_classes = 3 if config.env.cargo_classes % 3 == 0 else 2 # 2 weights for 2 classes, 3 weights for 3,6 classes
    config.env.capacity = [50] if config.env.TEU == 1000 else [500]
    return config

def make_env(env_kwargs, device):
    """Setup custom environment"""
    return MasterPlanningEnv(**env_kwargs).to(device)
    # return PortMasterPlanningEnv(**env_kwargs).to(device).half()

# def check_env_specs(env):
#     """Verifies that the environment's specifications (action and observation spaces) are valid."""
#     try:
#         action_spec = env.action_spec
#         observation_spec = env.observation_spec
#         print("Action space shape:", action_spec.shape)
#         print("Observation space shape:", observation_spec.shape)
#         return True
#     except AttributeError as e:
#         print(f"Error: {e}")
#         print("Please make sure your environment defines valid action_spec and observation_spec properties.")
#         return False

def main(config: Optional[DotMap] = None):
    # Initialize torch and cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64  # or some higher value

    # Set random seed and device
    torch.set_num_threads(1)
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
        torch.backends.cudnn.deterministic = True

    ## Environment initialization
    emb_dim = 128
    env_kwargs = config.env
    env = make_env(env_kwargs, device)

    print(env.batch_size)
    breakpoint()
    # check_env_specs(env)

    # env = DenseRewardTSPEnv(generator_params=dict(num_loc=100),)
    # env = SDVRPEnv(generator_params=dict(num_loc=100),)
    td = env.reset(batch_size=32)

    # Model: default is AM with REINFORCE and greedy rollout baseline
    # check out `RL4COLitModule` and `REINFORCE` for more details
    init_embedding = MPPInitEmbedding(emb_dim, env.action_spec.shape[0], env)
    context_embedding = MPPContextEmbedding(env.action_spec.shape[0], emb_dim, env)
    # policy = AttentionModelPolicy4PPO(env_name=env.name,
    policy = AttentionModelPolicy(env_name=env.name,
                                      encoder=AttentionModelEncoder(emb_dim, init_embedding=init_embedding,),
                                        decoder=AttentionModelDecoder(emb_dim, context_embedding=context_embedding, mask_inner=False),
                                  # this is actually not needed since we are initializing the embeddings!
                                  embed_dim=emb_dim,
                                      mask_inner=False,
                                      # train_decode_type="sampling",
                                      # val_decode_type="greedy",
                                      # test_decode_type="beam_search",
                                    train_decode_type="continuous_sampling",
                                    val_decode_type="continuous_projection",
                                    test_decode_type="continuous_projection",
                                    )

    # model = AttentionModel(
    model = AMPPO(
        env,
        policy=policy,
        # n_step = 100,
        # baseline="rollout",
        train_data_size=config.am_ppo.train_data_size,  #1_000_000; really small size for demo
        val_data_size=config.am_ppo.val_data_size,  #100_000
        policy_kwargs={  # we can specify the decode types using the policy_kwargs
            "train_decode_type": "continuous_sampling",
            "val_decode_type": "continuous_projection",
            "test_decode_type": "continuous_projection",
        }
    )

    # Example callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",  # save to checkpoints/
        filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
        save_top_k=1,  # save only the best model
        save_last=True,  # save the last model
        monitor="val/return",  # monitor validation reward
        mode="max",
    )  # maximize validation reward
    rich_model_summary = RichModelSummary(max_depth=3)  # model summary callback
    callbacks = [checkpoint_callback, rich_model_summary]

    # Logger
    # logger = WandbLogger(project="rl4co", name="tsp")
    logger = None # uncomment this line if you don't want logging

    # Main trainer configuration
    trainer = RL4COTrainer(
        max_epochs=2,
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        precision=16,
    )

    # Main training loop
    trainer.fit(model)

    # Greedy rollouts over trained model
    # note: modify this to load your own data instead!
    td_init = env.reset(batch_size=[16]).to(device)
    policy = model.policy.to(device)
    out = policy(
        td_init.clone(), env, phase="test", decode_type="greedy", return_actions=True
    )

    # Print results
    print(f"Tour lengths: {[f'{-r.item():.3f}' for r in out['reward']]}")
    print(f"Avg tour length: {-torch.mean(out['reward']).item():.3f}")


if __name__ == "__main__":
    # Load static configuration from the YAML file
    file_path = os.getcwd()
    with open(f'{file_path}/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    main(config)