import time
import torch
from models.decoding import rollout_mpp
from environment.data import StateDependentDataset, custom_collate_fn
from torch.utils.data import DataLoader
from tensordict import TensorDict

def trial(env, td, device, num_rollouts=30):
    """Run a trial of the environment"""

    # Run profiling on the environment
    # with torch.profiler.profile(
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    #         record_shapes=True,
    #         profile_memory=True
    # ) as prof:
    #     for idx in range(num_rollouts):
    #         td = env.reset(batch_size=td.batch_size, td=td)
    #         done = td["done"][0]
    #         while not done:
    #             td = random_action_policy(td)
    #             next_td = env.step(td)["next"]
    #             prof.step()  # Update the profiler
    #             td = next_td
    #             done = td["done"][0]

    # Test dataset
    batch_size = [1024]
    total_samples = 4_000_000
    dataset = StateDependentDataset(env, td, total_samples, batch_size)
    print(f"Number of batches: {len(dataset)}")

    # Test the environment
    def random_action_policy(td):
        """Helper function to select a random action from available actions"""
        batch_size = td.batch_size
        action = torch.distributions.Uniform(env.action_spec.low, env.action_spec.high).sample(batch_size) * 30
        td.set("action", action.to(torch.float16))
        return td

    # Rollout the environment with random actions
    runtimes_rollout = torch.zeros(num_rollouts, dtype=torch.float16, device=device)
    for idx in range(num_rollouts):
        with torch.cuda.amp.autocast():
            start_time = time.time()
            reward, td, actions = rollout_mpp(env, env.reset(batch_size=td.batch_size, td=td), random_action_policy)
            runtime = time.time() - start_time
            runtimes_rollout[idx] = runtime
    print("-"*50)
    print(f"Mean runtime: {torch.mean(runtimes_rollout):.3f} s")
    print(f"Std runtime: {torch.std(runtimes_rollout):.3f} s")
    print(f"Max runtime: {torch.max(runtimes_rollout):.3f} s")
    print(f"Min runtime: {torch.min(runtimes_rollout):.3f} s")
    print("-"*50)