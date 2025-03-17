import gc
import os
import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
# import matplotlib.pyplot as plt
import numpy as np

from clustering import GraphClustering
from data import dataset_reports, data_dir

from hgpsl_implementation.models import ModernizedHGPSL
from models import MCSFullModel

# Set up logging
def setup_logging(output_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )

def contrastive_loss(distances, labels, margin=5.0):

    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)
    loss = positive_loss + negative_loss

    return loss.mean()


def run_hgpsl_experiment(dataset_name,
                            train_num,
                            val_num,
                            test_num,
                            device='cpu',
                            run_name=None,
                            pretrain_epoch_num = None, dist_comp_timeout = 40):



    root_dir = './outputs/hgpsl_experiment_out'
    run_name = run_name or f'cluster_dataset_{time.strftime("%Y%m%d_%H%M%S")}'
    output_dir = Path(root_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)


    # Setup logging
    setup_logging(output_dir)
    logging.info(f"Starting experiment: {run_name}")

    # Create subdirectories for plots and results
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    results_dir = output_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    # Model paths
    original_model_path = output_dir / 'original_model'
    mcs_model_path = output_dir / 'mcs_model'

    dataset = TUDataset(root='./datasets/PROTEINS', name='PROTEINS', use_node_attr=True,
                        # pre_filter=lambda data: data.num_nodes <= 160,force_reload=True
                        ).shuffle()

    train_num = int(len(dataset) * 0.8)
    val_num = int(len(dataset) * 0.1)
    test_num = len(dataset) - (train_num + val_num)


    dataset_reports(dataset, path=str(output_dir / 'dataset_reports.txt'))

    train_dataset = dataset[:train_num]
    val_dataset = dataset[train_num:train_num + val_num]
    test_dataset = dataset[train_num + val_num:train_num + val_num + test_num]


    clustering = GraphClustering()
    wl_embeddings, cluster_labels = clustering.fit_predict(train_dataset)

    # settings
    batch_size = 30
    num_epochs = 10

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    train_loaders = []
    train_dict = {}
    for data, i in zip(train_dataset, cluster_labels):
        if not i in train_dict.keys():
            train_dict[i] = []
        train_dict[i].append(data)
    for i in train_dict.keys():
        train_loaders.append( DataLoader(train_dict[i], batch_size=batch_size, shuffle=True) )

    logging.info(f"Dataset size: {len(dataset)}, Train: {train_num}, Val: {val_num}, Test: {test_num}")
    logging.info(f"Device: {device}")

    in_channels = dataset.num_features
    num_classes = dataset.num_classes




    model_original = ModernizedHGPSL(in_channels,num_classes)
    model_mcs = MCSFullModel(in_channels,num_classes)

    model_original.to(device)
    model_mcs.to(device)

    # Optimizer and scheduler
    lr = 0.001

    optimizer_original = torch.optim.Adam(model_original.parameters(), lr=lr)  # , weight_decay=1e-5)
    # scheduler_metric = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    optimizer_mcs = torch.optim.Adam(model_mcs.parameters(), lr=lr)  # , weight_decay=1e-5)
    # scheduler_readout = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    # Tracking dictionaries
    metrics = {
        'MCS': {'train_loss': [], 'val_loss': [], 'val_acc': [], 'times': []},
        'Original': {'train_loss': [], 'val_loss': [], 'val_acc': [], 'times': []}
    }
    best_val_acc = {'MCS': 0, 'Original': 0}

    for epoch in range(num_epochs):
        logging.info(f"Starting Epoch {epoch + 1}/{num_epochs}")

        # Original model training
        start_time = time.time()
        train_loss, train_acc = train_original_model(model_original, optimizer_original, train_loader, device)
        val_acc, val_loss = test_original_model(model_original, val_loader, device)
        duration = time.time() - start_time

        metrics['Original']['train_loss'].append(train_loss)
        metrics['Original']['val_loss'].append(val_loss)
        metrics['Original']['val_acc'].append(val_acc)
        metrics['Original']['times'].append(duration)
        best_val_acc['Original'] = max(best_val_acc['Original'], val_acc)

        logging.info(f"Original Model - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {duration:.2f}s")

        # MCS model training
        start_time = time.time()
        train_loss, train_acc = train_mcs_model(model_mcs, optimizer_mcs, train_loaders, device, epoch)
        val_acc, val_loss = test_mcs_model(model_mcs, val_loader, device)
        duration = time.time() - start_time

        metrics['MCS']['train_loss'].append(train_loss)
        metrics['MCS']['val_loss'].append(val_loss)
        metrics['MCS']['val_acc'].append(val_acc)
        metrics['MCS']['times'].append(duration)
        best_val_acc['MCS'] = max(best_val_acc['MCS'], val_acc)

        logging.info(f"MCS Model - Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Time: {duration:.2f}s")

    # Test evaluation
    test_acc_orig, test_loss_orig = test_original_model(model_original, test_loader, device)
    test_acc_mcs, test_loss_mcs = test_mcs_model(model_mcs, test_loader, device)

    # Save results
    results = {
        'test_acc': {'Original': test_acc_orig, 'MCS': test_acc_mcs},
        'test_loss': {'Original': test_loss_orig, 'MCS': test_loss_mcs},
        'best_val_acc': best_val_acc,
        'metrics': metrics
    }
    np.save(results_dir / 'results.npy', results)

    # Generate and save plots
    plot_metrics(metrics, plot_dir)

    print(f"Test Accuracy - Original: {test_acc_orig:.4f}, MCS: {test_acc_mcs:.4f}")
    print(f"Best Val Accuracy - Original: {best_val_acc['Original']:.4f}, MCS: {best_val_acc['MCS']:.4f}")

def train_original_model(model, optimizer, train_loader , device):
    model.train()
    loss_train = 0.0
    correct = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    acc_train = correct / len(train_loader.dataset)

    return loss_train / len(train_loader.dataset), acc_train

def test_original_model(model, loader, device):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


def train_mcs_model(model, optimizer, train_loaders, device, epoch):
    model.train()
    loss_train = 0.0
    correct = 0
    train_size = 0
    batch_total = sum(len(loader) for loader in train_loaders)
    batch_count = 0

    if epoch > 5:
        model.freeze_pooling(True)
        model.pre_train = False
        logging.info("MCS Model: Switching from pre-training to full training")

    logging.info(f"MCS Training - Pre-train mode: {model.pre_train}")
    start_time = time.time()

    for loader_idx, train_loader in enumerate(train_loaders):
        train_size += len(train_loader.dataset)

        for batch_idx, data in enumerate(train_loader):
            batch_count += 1
            optimizer.zero_grad()
            data = data.to(device)

            batch_start = time.time()

            if model.pre_train:
                distances = model(data)
                labels = (data.y.view(-1, 1) == data.y.view(1, -1)).float()
                loss = contrastive_loss(distances, labels)
            else:
                out = model(data)
                loss = F.nll_loss(out, data.y)
                pred = out.max(dim=1)[1]
                correct += pred.eq(data.y).sum().item()

            loss.backward()
            optimizer.step()
            loss_train += loss.item()

            batch_time = time.time() - batch_start

            # Log every 5 batches or at the end of each loader
            if batch_count % 5 == 0 or batch_idx == len(train_loader) - 1:
                progress = batch_count / batch_total * 100
                running_loss = loss_train / batch_count
                logging.info(
                    f"MCS Progress: {progress:.1f}% "
                    f"(Loader {loader_idx + 1}/{len(train_loaders)}, "
                    f"Batch {batch_idx + 1}/{len(train_loader)}) "
                    f"- Loss: {running_loss:.4f}, "
                    f"Batch Time: {batch_time:.2f}s"
                )

    total_time = time.time() - start_time
    logging.info(f"MCS Training Complete - Total Time: {total_time:.2f}s")

    return loss_train / train_size, correct / train_size

def test_mcs_model(model, loader, device):

    if model.pre_train: return 0

    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


def plot_metrics(metrics, plot_dir):
    """Generate and save analysis plots"""
    epochs = range(len(metrics['Original']['train_loss']))

    # Loss plot
    plt.figure(figsize=(10, 6))
    for model in ['Original', 'MCS']:
        plt.plot(epochs, metrics[model]['train_loss'], label=f'{model} Train Loss')
        plt.plot(epochs, metrics[model]['val_loss'], label=f'{model} Val Loss', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'loss_plot.png')
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    for model in ['Original', 'MCS']:
        plt.plot(epochs, metrics[model]['val_acc'], label=f'{model} Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'accuracy_plot.png')
    plt.close()

    # Training time plot
    plt.figure(figsize=(10, 6))
    for model in ['Original', 'MCS']:
        plt.plot(epochs, metrics[model]['times'], label=f'{model} Time')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_dir / 'time_plot.png')
    plt.close()

    # Bar plot of average times
    plt.figure(figsize=(8, 6))
    avg_times = [np.mean(metrics[model]['times']) for model in ['Original', 'MCS']]
    plt.bar(['Original', 'MCS'], avg_times)
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Training Time per Epoch')
    plt.savefig(plot_dir / 'avg_time_bar.png')
    plt.close()