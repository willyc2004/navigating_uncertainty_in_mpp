import time
import torch

from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import WandbLogger

from rl4co.envs import TSPEnv, CVRPTWEnv, SDVRPEnv, DenseRewardTSPEnv, SPCTSPEnv
from rl4co.models.zoo import AttentionModel
from rl4co.utils.trainer import RL4COTrainer

from typing import Optional
import torch
import torch.nn as nn

from tensordict.tensordict import TensorDict
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    UnboundedContinuousTensorSpec,
    UnboundedDiscreteTensorSpec,
)

from rl4co.utils.decoding import rollout, random_policy
from rl4co.models.rl.ppo.stepwise_ppo import StepwisePPO
from rl4co.models.zoo import AttentionModel, AttentionModelPolicy, AMPPO
from rl4co.utils.ops import gather_by_index, get_tour_length
from rl4co.utils.trainer import RL4COTrainer

# RL4CO
from rl4co.utils.trainer import RL4COTrainer
from rl4co.models.zoo import AMPPO
# Customized RL4CO modules
from models.am_policy import AttentionModelPolicy4PPO
from models.projection_n_step_ppo import Projection_Nstep_PPO
from models.projection_ppo import Projection_PPO
from rl4co.models.zoo.am.encoder import AttentionModelEncoder
from models.examples.decoder2 import AttentionModelDecoder
from models.constructive import ConstructivePolicyMPP
from rl4co.models.common.constructive.autoregressive import AutoregressivePolicy
from environment.env import MasterPlanningEnv
# AMPPO.__bases__ = (StepwisePPO,)  # Adapt base class
# AMPPO.__bases__ = (Projection_Nstep_PPO,)  # Adapt base class
AMPPO.__bases__ = (Projection_PPO,)  # Adapt base class
from environment.embeddings import MPPContextEmbedding, MPPInitEmbedding

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

class TSPInitEmbedding(nn.Module):
    """Initial embedding for the Traveling Salesman Problems (TSP).
    Embed the following node features to the embedding space:
        - locs: x, y coordinates of the cities
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPInitEmbedding, self).__init__()
        node_dim = 2  # x, y

        self.init_embed = nn.Linear(node_dim, embed_dim, linear_bias)
        # input: xxx, node_dim --> xxx, obs_shape
        # output: xxx, embed_dim --> xxx, embed_dim
        # where xxx is the batch size, td["locs"]

    def forward(self, td):
        out = self.init_embed(td["locs"])
        return out

class TSPContext(nn.Module):
    """Context embedding for the Traveling Salesman Problem (TSP).
    Project the following to the embedding space:
        - first node embedding
        - current node embedding
    """

    def __init__(self, embed_dim, linear_bias=True):
        super(TSPContext, self).__init__()
        self.W_placeholder = nn.Parameter(
            torch.Tensor(2 * embed_dim).uniform_(-1, 1)
        )
        self.project_context = nn.Linear(
            embed_dim * 2, embed_dim, bias=linear_bias
        )

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        # By default, node_dim = -1 (we only have one node embedding per node)
        node_dim = (
            (-1,) if td["first_node"].dim() == 1 else (td["first_node"].size(-1), -1)
        )
        if td["i"][(0,) * td["i"].dim()].item() < 1:  # get first item fast
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            context_embedding = gather_by_index(
                embeddings,
                torch.stack([td["first_node"], td["current_node"]], -1).view(
                    batch_size, -1
                ),
            ).view(batch_size, *node_dim)
        output = self.project_context(context_embedding)
        return output

class StaticEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(StaticEmbedding, self).__init__()

    def forward(self, td):
        return 0, 0, 0

def check_env_specs(env):
    """Verifies that the environment's specifications (action and observation spaces) are valid."""
    try:
        action_spec = env.action_spec
        observation_spec = env.observation_spec
        print("Action space shape:", action_spec.shape)
        print("Observation space shape:", observation_spec.shape)
        return True
    except AttributeError as e:
        print(f"Error: {e}")
        print("Please make sure your environment defines valid action_spec and observation_spec properties.")
        return False

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
    # env = DenseRewardTSPEnv(generator_params=dict(num_loc=100),)
    # env = SDVRPEnv(generator_params=dict(num_loc=100),)
    check_env_specs(env)
    td = env.reset(batch_size=32)

    if env.name == "mpp":
        AutoregressivePolicy.__bases__ = (ConstructivePolicyMPP,)  # Adapt base class

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
                                  init_embedding=init_embedding,
                                      context_embedding=context_embedding,
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
        monitor="val/reward",  # monitor validation reward
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
    file_path = "/mnt/c/Users/jaiv/PycharmProjects/DRL_master_planning_problem/"
    with open(file_path + 'config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        config = DotMap(config)
        config = adapt_env_kwargs(config)

    main(config)