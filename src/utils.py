# medqa_project/src/utils.py
import torch

def get_device():
    """
    Determines and returns the available device (GPU if available, otherwise CPU).

    Returns
    -------
    torch.device
        The selected torch device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device