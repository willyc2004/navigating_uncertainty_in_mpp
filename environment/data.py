import torch
from torch.utils.data import Dataset, DataLoader, Sampler, IterableDataset
from tensordict import TensorDict

class StateDependentDataset(IterableDataset):
    """An iterable dataset that generates a batch of samples at iteration i conditioned on the batch at iteration i-1.

    Args:
        env: The environment object that contains the generator function.
        td: The state of the environment.
        total_samples: The total number of samples to generate.
        batch_size: The batch size to generate each iteration.
        device: The device to run the dataset on (GPU/CPU).
    """
    def __init__(self, env, td, total_samples, batch_size, device='cuda'):
        self.env = env
        self.generator = env.generator
        self.td = td
        self.total_samples = total_samples
        self.batch_size = batch_size
        self.num_batches = self.total_samples // self.batch_size  # Total number of batches
        self.device = device  # Set the device (GPU or CPU)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # First iteration or if the tensor dict is empty
        if self.td is None or self.td.is_empty():
            self.td = self.generator(self.batch_size).to(self.device)  # Move to GPU if necessary

        for _ in range(self.num_batches):
            # Generate updated batch conditioned on the previous state
            updated_td = self.generator(self.batch_size, self.td).to(self.device)  # Directly move to GPU
            yield updated_td  # Return the generated batch
            self.td = updated_td  # Update the state for the next iteration

def custom_collate_fn(batch):
    """Custom collate function that stacks data and moves to the GPU."""
    collated_data = {}

    for key in batch[0].keys():
        # Stack tensors for each key and ensure they are moved to the correct device
        collated_data[key] = torch.stack([item[key] for item in batch])

    key = list(collated_data.keys())[0]  # Get a random key to determine device and batch size
    # Use TensorDict and ensure the batch is on the correct device
    return TensorDict(collated_data, device=collated_data[key].device, batch_size=collated_data[key].shape[0])