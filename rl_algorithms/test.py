import time
import torch
from rl_algorithms.utils import make_env
from rl_algorithms.train import get_performance_metrics

def evaluate_model(policy, env_kwargs, device=torch.device("cuda"), num_episodes=10, n_step=100):
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
    # Create the test environment
    test_env = make_env(env_kwargs=env_kwargs, batch_size=[1], device=device)  # Single batch for evaluation
    test_env.eval()  # Set environment to evaluation mode
    policy.eval()  # Set policy to evaluation mode

    # Initialize metrics storage
    metrics = {
        "return": 0,
        "traj_return": 0,
        "total_violation": 0,
        "total_profit": 0,
        "inference_time": 0,
        "num_episodes": num_episodes,
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
            # Synchronize and measure inference time
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()

            # Perform rollout
            trajectory = test_env.rollout(
                policy=policy,
                max_steps=n_step,
                auto_reset=True,
            )

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.perf_counter()

            # Accumulate inference time for this episode
            metrics["inference_time"] += end_time - start_time

            # Compute performance metrics for the episode
            episode_metrics = get_performance_metrics(trajectory, trajectory, test_env)
            for key, value in episode_metrics.items():
                if key in metrics:
                    metrics[key] += value  # Aggregate metrics

    # Average metrics over the episodes
    for key in metrics:
        if key != "num_episodes":
            metrics[key] /= num_episodes

    # todo: add visualizations if needed
    test_env.close()
    return metrics
