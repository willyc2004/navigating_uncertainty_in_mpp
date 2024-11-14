from typing import Any, Dict, Generator, Optional, Type, TypeVar, Union, Tuple
import torch as th

# Transport sets
def get_transport_idx(P: int, device) -> Union[th.Tensor,]:
    # Get above-diagonal indices of the transport matrix
    origins, destinations = th.triu_indices(P, P, offset=1, device=device)
    return th.stack((origins, destinations), dim=-1)
# def get_transport_idx(P: Union[th.Tensor],) -> Union[th.Tensor,]:
#     # Get above-diagonal indices of the transport matrix
#     origins, destinations = th.triu_indices(P.squeeze(), P.squeeze(), offset=1, device=P.device)
#     return th.stack((origins, destinations), dim=-1)

def get_load_pods(POD: Union[th.Tensor]):
    # Get non-zero column indices
    return (POD > 0)

def get_load_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] == POL) & (transport_idx[:, 1] > POL)
    return mask

def get_discharge_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transports
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] == POL)
    return mask

def get_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] <= POL) & (transport_idx[:, 1] > POL)
    return mask

def get_not_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid cargo groups:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] >= POL)
    return mask

def get_remain_on_board_transport(transport_idx, POL) -> Union[th.Tensor]:
    # Boolean mask for valid transport:
    mask = (transport_idx[:, 0] < POL) & (transport_idx[:, 1] > POL)
    return mask

def get_pols_from_transport(transport_idx, P, dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POD
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 0].long()] = 1
    return one_hot

def get_pods_from_transport(transport_idx, P, dtype) -> Union[th.Tensor]:
    # Get transform array from transport to POD
    T = transport_idx.size(0)
    one_hot = th.zeros(T, P, device=transport_idx.device, dtype=dtype)
    one_hot[th.arange(T), transport_idx[:, 1].long()] = 1
    return one_hot

# Get step variables
def get_k_tau_pair(step, K):
    """Get the cargo class from the step number in the episode
    - step: step number in the episode
    - T: number of transports per episode
    """
    k = step % K
    tau = step // K
    return k, tau

def get_pol_pod_pair(tau, P):
    """Get the origin-destination (pol,y) pair of the transport with index i
    - i: index of the transport
    - P: number of ports
    - pol: origin
    - pod: destination
    """
    # Calculate pol using the inverse of triangular number formula
    ## todo: check if this is formulation is correct for P!=4. (empirically it seems to work)
    pol = P - 2 - th.floor(th.sqrt(2*(P*(P-1)//2 - 1 - tau) + 0.25) - 0.5).to(th.int64)
    # Calculate y based on pol
    pod = tau - (P*(P-1)//2 - (P-pol)*(P-pol-1)//2) + pol + 1
    return pol, pod

def get_transport_from_pol_pod(pol, pod, transport_idx):
    """Get the transport index from the origin-destination pair
    - pol: origin
    - pod: destination
    - transport_idx: transport tensor to look up row that matches the origin-destination pair
    """
    # Find rows where both the first column is `pol` and the second column is `pod`
    mask = (transport_idx[:, 0].unsqueeze(1) == pol) & (transport_idx[:, 1].unsqueeze(1) == pod)
    # Use torch.where to get the indices where the mask is True
    output = th.where(mask)[0] # [0] extracts the first dimension (row indices)

    # Check if the output is empty
    if output.numel() == 0:
        return th.tensor([0], device=transport_idx.device)

    return output

def compute_violation(lhs_A, rhs, action,) -> th.Tensor:
    """Compute violations and loss of compact form"""
    lhs = (lhs_A * action).sum(dim=(-1))
    output = th.clamp(lhs-rhs, min=0)
    return output

if __name__ == "__main__":
    # Test the transport sets
    print(get_pol_pod_pair(tau=th.tensor(7), P=th.tensor(5)))
