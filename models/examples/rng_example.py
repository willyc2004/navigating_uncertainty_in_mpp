import torch

class RandomMatrixGenerator:
    def __init__(self, seed: int, device: str = "cpu"):
        """
        Initialize a random matrix generator with a specific seed.

        Args:
            seed (int): The seed for reproducibility.
            device (str): The device to use for the generator.
        """
        self.seed = seed
        self.device = device
        self.rng = torch.Generator(device=device).manual_seed(seed)

    def generate_matrix(self, shape: tuple):
        """
        Generate a reproducible random matrix.

        Args:
            shape (tuple): The shape of the matrix.

        Returns:
            torch.Tensor: A random matrix.
        """
        return torch.randn(shape, generator=self.rng, device=self.device)

# Example Usage
seed = 42
shape = (3, 3)
device = "cuda"

# Initialize the generator
matrix_generator = RandomMatrixGenerator(seed, device)

# Generate a matrix
matrix1 = matrix_generator.generate_matrix(shape)
print("First matrix:\n", matrix1)

# Generate another matrix with the same generator
matrix2 = matrix_generator.generate_matrix(shape)
print("Second matrix:\n", matrix2)

# Reinitialize with the same seed and regenerate the first matrix
matrix_generator_reinitialized = RandomMatrixGenerator(seed, device)
matrix1_reproducible = matrix_generator_reinitialized.generate_matrix(shape)
print("Reproducible first matrix:\n", matrix1_reproducible)

# Confirm the reproducibility
assert torch.equal(matrix1, matrix1_reproducible), "Matrices should match for the same seed!"
