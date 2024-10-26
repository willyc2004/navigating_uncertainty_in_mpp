# DRL_4_Master_Planning

**Deep Reinforcement Learning for Master Planning Problem (MPP)**

A Deep Reinforcement Learning (DRL) framework for constructing solutions to the Master Planning Problem (MPP) in Container Vessel Stowage Planning. The MPP is a combinatorial optimization problem that aims to find a global stowage plan on a container vessel, maximizing total profit during a fixed-schedule multi-port voyage. For more information, see the [Literature survey on the container vessel stowage planning problem](https://www.sciencedirect.com/science/article/pii/S0377221723009517?via%3Dihub).

The goal of this framework is to provide researchers and practitioners with an environment to benchmark solution methods on this sequential decision problem.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Adjustments to RL4CO Codebase](#adjustments-to-rl4co-codebase)
- [License](#license)
- [Contact](#contact)

## Introduction

The **DRL_4_Master_Planning** project provides a Deep Reinforcement Learning-based approach to solve the Master Planning Problem for container vessel stowage. Built upon the [RL4CO](https://github.com/ai4co/rl4co) framework, it introduces several novel modifications, including continuous action spaces and projection-based constraints for feasibility.

### Features:
- Multi-head attention mechanism and PPO for reinforcement learning.
- Continuous action decoding strategies with action projection.
- Supports CUDA for GPU acceleration and integration with CPLEX for optimization tasks.
- Integration with Weights and Biases (WandB) for experiment tracking.

## Installation

### Prerequisites
To run the project locally, ensure you have the following:
- **Python 3.9+**
- **CUDA 12.2** (for GPU support)
- **CPLEX solver** (optional, for optimization tasks)
- **WandB** (optional, for experiment tracking)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.itu.dk/jaiv/DRL_4_master_planning.git
   cd DRL_4_master_planning
   ```

2. **Install dependencies:**

   Update pip and install the required dependencies:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

3. **Setup Weights and Biases (Optional):**

   If you are using [WandB](https://docs.wandb.ai/quickstart) for experiment tracking, log in to your WandB account:

   ```bash
   wandb login
   ```

   Follow the link in the terminal to authenticate.

## Usage

Once you have installed all the dependencies, you can train or test models.

### Training

To train a model, you can run:

```bash
python train.py --config config/train_config.yaml
```

### Testing

To evaluate a pretrained model, use:

[//]: # (```bash)

[//]: # (python test.py --model-path <path_to_model> --config config/test_config.yaml)

[//]: # (```)

## Code Structure

The framework is based on the [RL4CO](https://github.com/ai4co/rl4co) repository and contains the following modules:

### **Environment Modules**
- **env.py**: Contains the environment class for the MPP.
- **generator.py**: Contains the generator class for the MPP.
- **data.py**: Contains the dataset class for the MPP.
- **embeddings.py**: Contains the embedding classes for the MPP.
- **utils.py**: Contains utility functions for the MPP.

### **Model Modules**
- **decoder.py**: Contains the decoder classes for the model.
- **projection.py**: Contains the projection layer classes.
- **decoding.py**: Contains the decoding strategy classes.
- **constructive.py**: Contains the constructive policy class.
- **ppo.py**: Implements the Proximal Policy Optimization (PPO) algorithm.

## Adjustments to RL4CO Codebase

This project builds on RL4CO by introducing several key modifications:

- **decoding.py:**
  - Enables continuous decoding strategies.
  - Implements clipped Gaussian policy for continuous actions.
  - Projects actions into feasible space.
  
- **ppo.py:**
  - Incorporates learning based on feasibility and projection losses.
  
- **constructive.py:**
  - Allows for passing supportive data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any inquiries, issues, or feature requests, please contact:

- **Jaike van Twiller** - [jaiv@itu.dk](mailto:jaiv@itu.dk)