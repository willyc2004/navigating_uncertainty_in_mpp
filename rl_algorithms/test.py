import time
import torch
import math
from rl_algorithms.utils import make_env
from rl_algorithms.train import get_performance_metrics

# Functions
def get_z_score_torch(confidence_level):
    """
    Compute the z-score for a given confidence level using PyTorch.
    Args:
        confidence_level (float): Confidence level (e.g., 0.95 for 95% confidence).
    Returns:
        float: The z-score corresponding to the confidence level.
    """
    alpha = 1 - confidence_level
    return torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2)).item()

def compute_summary_stats(metrics, confidence_level=0.95):
    """
    Compute mean, median, std, min, and max for each metric in the dictionary.
    Args:
        metrics (dict): Dictionary with metrics as tensors.
    Returns:
        dict: Summary statistics for each metric.
    """
    summary_stats = {}
    z = get_z_score_torch(confidence_level)
    for key, values in metrics.items():
        n = values.numel()  # Number of elements in the tensor
        mean = values.mean().item()
        std = values.std().item()
        margin_of_error = z * (std / math.sqrt(n)) if n > 1 else 0.0  # CI margin of error

        summary_stats[key] = {
            "mean": mean,
            "median": values.median().item(),
            "std": std,
            "min": values.min().item(),
            "max": values.max().item(),
            "lb_ci": mean - margin_of_error,  # Lower bound of the CI
            "ub_ci": mean + margin_of_error,  # Upper bound of the CI
        }
    return summary_stats

# Main function
def evaluate_model(policy, config, device=torch.device("cuda"), **kwargs):
    """
    Evaluate the policy and critic on the test environment.

    Args:
        policy (ProbabilisticActor): The policy to evaluate.
        env_kwargs (dict): Environment configuration.
        device (torch.device): Device to perform computations.
        num_episodes (int): Number of episodes for evaluation.
        n_step (int): Maximum steps per episode.

    Returns:
        dict: A dictionary of evaluation metrics.
    """
    # Extract evaluation hyperparameters
    env_kwargs = config.env
    batch_size = config.model.get("batch_size", 2)
    num_episodes = kwargs.get("num_episodes", 10)

    # Create the test environment # todo: error with batch_size [1]
    test_env = make_env(env_kwargs=env_kwargs, batch_size=[batch_size], device=device)  # Single batch for evaluation
    n_step = test_env.T * test_env.K  # Maximum steps per episode (T x K)
    test_env.eval()  # Set environment to evaluation mode
    policy.eval()  # Set policy to evaluation mode

    # Initialize metrics storage
    metrics = {
        "total_profit": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "total_violations": torch.zeros(num_episodes, device=device),  # [num_episodes]
        "inference_times": torch.zeros(num_episodes, device=device),  # [num_episodes]
    }

    with torch.no_grad():
        # Warm-up phase
        for _ in range(10):
            _ = test_env.rollout(
                policy=policy,
                max_steps=n_step,
                auto_reset=True,
            )

        # Evaluation phase
        for episode in range(num_episodes):
            # Test on different seeds
            test_env.set_seed(episode + 1 + config.env.seed)

            # Synchronize and measure inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()

            # Perform rollout
            trajectory = test_env.rollout(
                policy=policy,
                max_steps=n_step,
                auto_reset=True,
            )

            # Synchronize and measure inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.perf_counter()

            # Extract episode-level metrics
            metrics["total_profit"][episode] = trajectory["profit"].mean(dim=0).sum()
            metrics["total_violations"][episode] = trajectory["violation"].mean(dim=0).sum()
            metrics["inference_times"][episode] = end_time - start_time



            # Print to analyze
            demand_violations = trajectory["action"].mean(dim=0).sum(dim=-1) - trajectory["observation"]["realized_demand"][:,0,].mean(dim=0)
            backorders = trajectory["observation"]["realized_demand"][:,0,].mean(dim=0) - trajectory["action"].mean(dim=0).sum(dim=-1)


            max_demand = trajectory["observation"]["realized_demand"].max().clamp(max=test_env.generator.train_max_demand)
            print("max_demand", max_demand, "train_max_demand", test_env.generator.train_max_demand)
            print("real_demand(mean,std)", trajectory["observation"]["realized_demand"].mean(), trajectory["observation"]["realized_demand"].std())
            print("realized_demand", trajectory["observation"]["realized_demand"][:,0,].mean(dim=0))
            print("norm_realized_demand", trajectory["observation"]["realized_demand"][:, 0,].mean(dim=0) / trajectory["observation"]["realized_demand"].max())
            print("action", trajectory["action"][0].sum(dim=-1))
            print("demand_viol", demand_violations.clamp(min=0).sum())
            print("backorders", backorders.clamp(min=0).sum())

    # Summarize episode-level metrics (mean and std)
    summary_stats = compute_summary_stats(metrics)

    test_env.close()
    return metrics, summary_stats
