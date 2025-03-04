import os
import logging
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures, Constant
from torch_geometric.datasets import TUDataset
import random

data_dir = '../datasets'


def dataset_reports(dataset, verbose=True, path=None):
    report = [f'Dataset: {dataset}', '====================', f'Number of graphs: {len(dataset)}',
              f'Number of features: {dataset.num_features}', f'Number of classes: {dataset.num_classes}']

    # Information about the dataset

    # Information about the first graph in the dataset
    data = dataset[0]
    report.append(f'First graph details: {data}')

    # Statistics about the dataset
    num_nodes_list = [data.num_nodes for data in dataset]
    num_edges_list = [data.num_edges for data in dataset]
    mean_features_list = [data.x.mean(dim=0) for data in dataset]

    report.append(f'Average number of nodes: {sum(num_nodes_list) / len(num_nodes_list)}')
    report.append(f'Average number of edges: {sum(num_edges_list) / len(num_edges_list)}')
    report.append(f'Average node features: {sum(mean_features_list) / len(mean_features_list)}')

    classes = {}
    for data in dataset:
        if data.y.item() not in classes:
            classes[data.y.item()] = 1
        else:
            classes[data.y.item()] += 1

    report.append(f'Class frequency: {classes}')
    # Print to console if verbose is True
    if verbose:
        for line in report:
            print(line)

    # Log to file
    if path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Open the file in write mode ('w') or append mode ('a')
        with open(path, 'w') as file:
            for line in report:
                file.write(line + '\n')

        # for line in report:
        #     logging.info(line)

    return report, classes


def load_dataset(name, verbose=True, *args):
    if name == 'MUTAG':
        dataset = TUDataset(root='datasets/MUTAG', name='MUTAG')
    elif name == 'ENZYMES':
        dataset = TUDataset(root='datasets/ENZYMES', name='ENZYMES')
    elif name == 'PROTEINS':
        dataset = TUDataset(root='datasets', name='PROTEINS', use_node_attr=True)
    elif name == 'COX2':
        dataset = TUDataset(root='datasets/COX2', name='COX2')
    elif name == 'Letter-high':
        dataset = TUDataset(root='datasets/Letter-high', name='Letter-high', use_node_attr=True)
    elif name == 'Letter-low':
        dataset = TUDataset(root='datasets/Letter-low', name='Letter-low', use_node_attr=True)
    elif name == 'TRIANGLES':
        dataset = TUDataset(root='datasets/TRIANGLES', name='TRIANGLES', transform=Constant(1))
    elif name == 'IMDB-MULTI':
        dataset = TUDataset(root='datasets/IMDB-MULTI', name='IMDB-MULTI', transform=Constant(1))
    else:
        raise ValueError("invading dataset name")

    if verbose:
        dataset_reports(dataset)

    return dataset


def get_dataloaders(dataset, train_num, val_num, test_num=0):
    # Ensure the total number of requested samples does not exceed the dataset size
    total_requested = train_num + val_num + test_num
    if total_requested > len(dataset):
        raise ValueError("Requested more samples than available in the dataset")

    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle()
    train_dataset = shuffled_dataset[:train_num]
    val_dataset = shuffled_dataset[train_num:train_num + val_num]
    test_dataset = shuffled_dataset[train_num + val_num:train_num + val_num + test_num]

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False) if train_num > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) if val_num > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) if test_num > 0 else None

    return train_loader, val_loader, test_loader


def balanced_subset(dataset, m, lables=None):
    # Create a dictionary to store m graphs per class for the training set
    label_dict = {}
    # train_data = []
    train_indices = list()

    if lables is None:
        for data in dataset:
            label = data.y.item()
            if label not in label_dict:
                label_dict[label] = []
    else:
        for lable in lables:
            label_dict[label] = []

    # Create the balanced training set
    for idx, data in enumerate(dataset):
        label = data.y.item()

        if label not in label_dict: continue

        if len(label_dict[label]) < m:
            label_dict[label].append(data)
            # train_data.append(data)
            train_indices.append(idx)

        # Stop if we have m samples for each label
        if all(len(samples) >= m for samples in label_dict.values()):
            break
    return train_indices


def get_datasubsets(dataset, train_num, val_num, test_num=0, each_class_train=None, each_class_val=None):
    # Ensure the total number of requested samples does not exceed the dataset size
    total_requested = 0

    total_requested = total_requested + each_class_train * dataset.num_classes if each_class_train is not None else total_requested + train_num

    total_requested = total_requested + each_class_val * dataset.num_classes if each_class_train is not None else total_requested + val_num

    total_requested = total_requested + test_num

    if total_requested > len(dataset):
        raise ValueError("Requested more samples than available in the dataset")

    # Shuffle and split the dataset
    shuffled_dataset = dataset.shuffle()

    if each_class_train is not None:
        train_indices = balanced_subset(dataset, each_class_train)
    else:
        train_indices = range(train_num)

    remaining_indices = [i for i in range(len(dataset)) if i not in train_indices]

    if each_class_val is not None:
        val_indices = balanced_subset(dataset[remaining_indices], each_class_val)
    else:
        val_indices = random.choice(remaining_indices, val_num)

    remaining_indices = [i for i in remaining_indices if i not in val_indices]

    test_indices = random.choice(remaining_indices, test_num) if test_num != 0 else None

    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    test_dataset = None if test_indices is None else dataset[test_indices]

    return train_dataset, val_dataset, test_dataset