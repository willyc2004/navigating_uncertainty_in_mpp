import torch

# Load the .pth file
file_path = "20250114_060653_/policy.pth"  # Replace with the actual path
state_dict = torch.load(file_path)

# Check for NaNs in parameters
nan_found = False
for name, param in state_dict.items():
    if isinstance(param, torch.Tensor):  # Ensure it's a tensor
        if torch.isnan(param).any():
            print(f"NaN found in {name}")
            nan_found = True

if not nan_found:
    print("No NaNs found in the saved model.")