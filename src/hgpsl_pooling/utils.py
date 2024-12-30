import logging
import os

import torch
from matplotlib import pyplot as plt
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

from data import data_dir

def debug_tensor(name, tensor):
    """Log tensor details like shape, dtype, and device."""
    if tensor is not None:
        logging.info(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
    else:
        logging.info(f"{name}: None")


def save_model(model, optimizer, epoch, path):
    """Save model and optimizer state."""
    path += '.pt'
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, path)
    logging.info(f"Model saved to {path}")


def load_model(model, optimizer, path, device):
    """Load model and optimizer state."""
    path += '.pt'
    if os.path.exists(path):
        state = torch.load(path, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        logging.info(f"Model loaded from {path}, starting from epoch {start_epoch}")
        return start_epoch
    else:
        logging.warning(f"No checkpoint found at {path}, starting from scratch")
        return 1


def save_results_report(results, path):
    """Save results to a text file."""
    with open(path, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    logging.info(f"Results report saved to {path}")


def save_plot(data, title, xlabel, ylabel, save_path):
    """Generate and save a plot."""
    plt.figure()
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(save_path)
    logging.info(f"Plot saved to {save_path}")
    plt.close()





