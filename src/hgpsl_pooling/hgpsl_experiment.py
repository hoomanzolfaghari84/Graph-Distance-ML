import gc
import os
import logging
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from data import dataset_reports, data_dir
from hgpsl_pooling.models import MCS_HGPSL, HPGSL_Original_Model

from hgpsl_pooling.utils import load_model, save_model, save_results_report, save_plot

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def contrastive_loss(distances, labels, margin=5.0):
    """
    Compute contrastive loss for graph distance pairs.

    Args:
        distances (torch.Tensor): Pairwise distances (batch_size, batch_size).
        labels (torch.Tensor): Pairwise labels (1 for similar, 0 for dissimilar).
        l (float): Pooling link loss.
        e (float): Pooling entropy loss.
        margin (float): Margin for dissimilar pairs.
        alpha (float): Pooling link loss impact factor.
        beta (float): Pooling entropy loss impact factor.

    Returns:
        torch.Tensor: Contrastive loss value.
    """

    positive_loss = labels * distances.pow(2)
    negative_loss = (1 - labels) * F.relu(margin - distances).pow(2)
    loss = positive_loss + negative_loss
    return loss.mean()


def run_hgp_sl_experiment(dataset_name,
                           train_num,
                             val_num,
                               test_num, 
                               device='cpu',
                                 run_name=None,
                                   pretrain_epoch_num = None, dist_comp_timeout = 40):
    """
        Run the pooling experiment and train the models.

        Args:
            dataset_name (str): Name of the dataset.
            train_num (int): Number of training samples.
            val_num (int): Number of validation samples.
            test_num (int): Number of test samples.
            device (str): Device to use ('cpu' or 'cuda').
            run_name (str): Optional experiment name for saving results.

        Returns:
            None
    """
    root_dir = './outputs/hgp_sl_experiment_out'
    
    # Define experiment directory
    if run_name:
        root_dir = os.path.join(root_dir, run_name)
    os.makedirs(root_dir, exist_ok=True)  # Set this to False immediately so we don't mistakenly erase last results


    # Model paths
    model_metric_path = os.path.join(root_dir, 'model_metric')
    model_metric_finetune_path = os.path.join(root_dir, 'model_metric_f')
    model_readout_path = os.path.join(root_dir, 'model_readout')
    results_path = os.path.join(root_dir, 'results.txt')

    ## Dataset and loader setup

    dataset = TUDataset(root='./datasets/PROTEINS', name='PROTEINS', use_node_attr=True,
                        # pre_filter=lambda data: data.num_nodes <= 160,force_reload=True
                        ).shuffle()
    
    
    if train_num is None or val_num is None or test_num is None :   
        train_num = int(len(dataset) * 0.8)
        val_num = int(len(dataset) * 0.1)
        test_num = len(dataset) - (train_num + val_num)


    dataset_reports_path = os.path.join(root_dir, 'dataset_reports.txt')
    dataset_reports(dataset, path=dataset_reports_path)

    
    # i = 0
    # j = 0
    # for data in dataset:
    #     if len(data.x) < 10:
    #         i+=1

    #         if len(data.edge_index[0]) > 20:
    #             j += 1

    # print(f'more than nodes: {i}')
    # print(f'more than edge index: {j}')


    # return

    train_dataset = dataset[:train_num]
    val_dataset = dataset[train_num:train_num + val_num]
    test_dataset = dataset[train_num + val_num:train_num + val_num + test_num]

    logging.info(f"Train num: {len(train_dataset)}")
    logging.info(f"Val num: {len(val_dataset)}")
    logging.info(f"Test num: {len(test_dataset)}")

    with open(dataset_reports_path,'a') as tmp:
        tmp.write(f"Train num: {len(train_dataset)}\n")
        tmp.write(f"Val num: {len(val_dataset)}")
        tmp.write(f"Test num: {len(test_dataset)}")

    batch_size = 30
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#,pin_memory_device=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    in_channels = dataset.num_features

    class Args:
        num_features = in_channels
        nhid = 128
        num_classes = dataset.num_classes
        pooling_ratio = 0.5
        dropout_ratio = 0.0
        sample_neighbor = True
        sparse_attention = True
        structure_learning = True
        lamb = 1.0

    args = Args()
    args_mcs = Args()

    model_metric = MCS_HGPSL(args_mcs, dist_comp_timeout=dist_comp_timeout)
    model_readout = HPGSL_Original_Model(args)

    model_metric.to(device)
    model_readout.to(device)

    # Optimizer and scheduler
    lr = 0.001
    optimizer_metric = torch.optim.Adam(model_metric.parameters(), lr=lr)  # , weight_decay=1e-5)
    # scheduler_metric = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    optimizer_readout = torch.optim.Adam(model_readout.parameters(), lr=lr)  # , weight_decay=1e-5)
    # scheduler_readout = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_metric, 'max', factor=0.1, patience=5)

    # Load models and optimizers from checkpoint
    start_epoch_metric = load_model(model_metric, optimizer_metric, model_metric_path, device)
    start_epoch_readout = load_model(model_readout, optimizer_readout, model_readout_path, device)

    models = {'Metric': model_metric, 'Readout': model_readout}
    optimizers = {'Metric': optimizer_metric, 'Readout': optimizer_readout}
    # schedulers = {'Metric': scheduler_metric, 'Readout': scheduler_readout}
    epoch_losses = {'Metric': [], 'Readout': []}
    best_val_acc = {'Metric': 0, 'Readout': 0}
    test_acc = {'Metric': 0, 'Readout': 0}
    times = {'Metric': [], 'Readout': []}
    val_accuracies = {'Metric': [], 'Readout': []}
    num_epochs = 16

    if pretrain_epoch_num is None:
        pretrain_epoch_num = num_epochs/2

    epoch_results = {}
    test_results = None
    # max(start_epoch_metric, start_epoch_readout)
    for epoch in range(0, num_epochs):
        for name, model in models.items():
            logging.info(f"Training {name} Model for Epoch {epoch}:")

            start = time.time()
            if epoch == pretrain_epoch_num + 1 and name == 'Metric':
                model.set_fine_tuning(True, freeze_pooling=True)
                best_val_acc['Metric'] = 0

            if epoch <= pretrain_epoch_num and name == 'Metric':
                train_loss = train_metric(epoch, model=model, optimizer=optimizers[name],
                                          train_loader=train_loader, device=device)
            else:
                train_loss, train_acc = train(model, train_loader, optimizers[name], device)


            epoch_losses[name].append(train_loss)
            with torch.no_grad():

                if epoch <= pretrain_epoch_num and name == 'Metric':
                    val_acc, _ = 0, 0
                    val_accuracies[name].append(val_acc)

                    save_model(model, optimizers[name], epoch,
                               model_metric_path)

                else:
                    logging.info(f"Validating")
                    val_acc, val_loss = test(val_loader, model, device)
                    val_accuracies[name].append(val_acc)

                    # schedulers[name].step(val_acc)
                    if val_acc > best_val_acc[name]:
                        logging.info("Testing")
                        test_acc[name], test_loss = test(test_loader, model, device)
                        best_val_acc[name] = val_acc

                        save_model(model, optimizers[name], epoch,
                                   model_metric_finetune_path if name == 'Metric' else model_readout_path)
                # logging.info(
                #     f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                #     f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f}'
                # )
                logging.info(
                    f'Epoch: {epoch:03d}, Model: {name}, Train loss: {train_loss:.4f} Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f}')

                times[name].append(time.time() - start)
                logging.info(f"Median Time per Epoch for {name}: {torch.tensor(times[name]).median():.4f}s")
                epoch_results[
                    f'Epoch: {epoch:03d}, Model: {name}'] = f'Train loss: {train_loss:.4f} Val Acc: {val_acc:.4f}, Test Acc: {test_acc[name]:.4f} Median time: {torch.tensor(times[name]).median():.4f}s'
                print(f"Median time: {torch.tensor(times[name]).median():.4f}s")
                print("=====================================================================")

    # Save results and plots
    results = {
        'Dataset': dataset_name,
        'Best Validation Accuracy': best_val_acc,
        'Test Accuracy': test_acc,
        'Median Time per Epoch': {name: torch.tensor(times[name]).median().item() for name in times}
    }
    try:
        results.update(epoch_results)
    except Exception as e:
        print(f"Error {e}")

    save_results_report(results, results_path)
    save_plot(epoch_losses['Metric'], "Metric Model Loss", "Epoch", "Loss", os.path.join(root_dir, "metric_loss.png"))
    save_plot(epoch_losses['Readout'], "Readout Model Loss", "Epoch", "Loss",
              os.path.join(root_dir, "readout_loss.png"))
    save_plot(val_accuracies['Metric'], "Metric Model Validation Accuracy", "Epoch", "Accuracy",
              os.path.join(root_dir, "metric_val_acc.png"))
    save_plot(val_accuracies['Readout'], "Readout Model Validation Accuracy", "Epoch", "Accuracy",
              os.path.join(root_dir, "readout_val_acc.png"))


def train(model, train_loader, optimizer, device):
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


def test(loader, model, device):
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


def train_metric(epoch, model, optimizer, train_loader: DataLoader, device, scheduler=None):
    model.train()
    loss_all = 0
    inters_num = 0
    losses = []

    for batch_idx, data in enumerate(train_loader):

        # xs, adjs, ys = data.x, data.adj, data.y
        data.to(device)

        # print(f'data {data}')
        # print(data)

        optimizer.zero_grad()

        distances = model(data)
        labels = (data.y.view(-1, 1) == data.y.view(1, -1)).float()
        loss = contrastive_loss(distances, labels)

        # loss = F.nll_loss(output, data.y.view(-1))

        loss.backward()

        # loss_all += data.y.size(0) * float(loss)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()


        loss = loss.clone().detach().cpu()
        losses.append(loss)
        loss_all += loss.item()

        del loss
        del distances
        del data

        gc.collect()

        inters_num += 1

        with torch.no_grad():
            # logging.info(f"Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}")
            if inters_num % 5 == 0:
                logging.info(f"====> Loss Mean: {np.mean(losses)}, Loss Var {np.var(losses)}")

    if scheduler is not None:
        # Step the scheduler after the epoch
        scheduler.step()

    logging.info(f"Epoch:{epoch}, processed {inters_num} batches. epoch train done")

    return loss_all / len(train_loader.dataset)

