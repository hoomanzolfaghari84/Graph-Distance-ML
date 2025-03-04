import torch
from grakel.kernels import WeisfeilerLehman, VertexHistogram
import grakel
import networkx as nx
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from torch_geometric.utils import to_dense_batch, to_dense_adj, to_networkx, subgraph


def pyg_to_grakel_graph(data):
    """
    Converts a PyG graph (data object) directly into a grakel.Graph.

    Parameters:
    -----------
    data : torch_geometric.data.Data
        A PyG graph data object.

    Returns:
    --------
    G_grakel : grakel.Graph
        A GraKeL graph with the edge list as the initialization object and a node label dictionary.
    """

    # Extract the edge_index as a numpy array (shape [2, num_edges])
    if isinstance(data, tuple):
        edge_index = data[1].cpu().numpy()  # ensure on CPU
        x = data[0]
        num_nodes = len(x)
    else:
        edge_index = data.edge_index.cpu().numpy()  # ensure on CPU
        num_nodes = data.num_nodes
        x = data.x

    # Create a set of edges (as tuples) to avoid duplicate edges in the undirected graph.
    # Note: We sort the tuple (min, max) so that (u, v) and (v, u) become the same edge.
    edges = set()
    for src, dst in edge_index.T:
        edge = (int(src), int(dst))
        # For undirected graphs, ensure each edge is added only once (by ordering the nodes)
        edge = tuple(sorted(edge))
        edges.add(edge)
    # Convert the set to a list
    edge_list = list(edges)

    # Build node labels: keys: node indices, values: label
    # If features exist, convert feature vector to tuple; otherwise, use a default label of 0.

    node_labels = {}
    if x is not None:
        # data.x is assumed to be a tensor of shape [num_nodes, num_features].
        # Depending on your needs, you might discretize or otherwise process these features.
        for i in range(num_nodes):
            node_labels[i] = tuple(x[i].cpu().tolist())
    else:
        for i in range(num_nodes):
            node_labels[i] = 0

    # Create the grakel.Graph.
    # The initialization_object here is the edge list.
    G_grakel = grakel.Graph(edge_list, node_labels=node_labels)

    return G_grakel


def pyg_data_to_nx(data):
    """
    Converts a PyG data object to a NetworkX graph.
    This function extracts the edge_index and (optionally) node features as node labels.
    """
    G = nx.Graph()
    num_nodes = data.num_nodes
    # if node features exist and can be used as labels:
    if data.x is not None:
        # In this example we use the argmax of features or a simple transformation as a label.
        # Adjust according to your data. Here, we simply convert the feature tensor to a tuple.
        for i in range(num_nodes):
            # For simplicity, we convert the feature vector to a tuple (or you could use a discretization)
            G.add_node(i, label=tuple(data.x[i].tolist()))
    else:
        for i in range(num_nodes):
            G.add_node(i, label=0)

    # Add edges (convert PyG's edge_index to edge list)
    edge_index = data.edge_index.numpy()
    # PyG edge_index is of shape [2, num_edges]. We add each edge (assume undirected graphs)
    for src, dst in edge_index.T:
        G.add_edge(src, dst)
    return G


class GraphClustering:

    def __init__(self, n_clusters=20):
        # Use Kernel PCA to project the data into a Euclidean space
        self.kpca = KernelPCA(n_components=10, kernel="precomputed", random_state=42)

        # Then perform k-means clustering on the embeddings
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Define the WL kernel with n_iters iterations
        self.wl_kernel = WeisfeilerLehman(n_iter=5, normalize=True, base_graph_kernel=VertexHistogram)

    def __fit_kernel__(self, dataset):
        # Convert each graph in the PyG dataset to a NetworkX graph.
        print("Converting PyG graphs to NetworkX graphs...")
        graphs_grakel = [pyg_to_grakel_graph(data, ) for data in dataset]
        # graphs_gk = grakel.graph_from_networkx(graphs_nx)
        data_lables = [data.y for data in dataset]
        print("Converted {} graphs.".format(len(graphs_grakel)))

        # Compute the kernel matrix for the list of graphs (NetworkX graphs are acceptable for GraKeL)
        K = self.wl_kernel.fit_transform(graphs_grakel)
        print("WL Kernel matrix shape:", K.shape)
        return K, data_lables

    def __transform_kernel__(self, dataset):
        # Convert each graph in the PyG dataset to a NetworkX graph.
        print("Converting PyG graphs to NetworkX graphs...")
        graphs_grakel = [pyg_to_grakel_graph(data, ) for data in dataset]
        # graphs_gk = grakel.graph_from_networkx(graphs_nx)
        # data_lables = [data[2] for data in dataset]
        print("Converted {} graphs.".format(len(graphs_grakel)))

        # Compute the kernel matrix for the list of graphs (NetworkX graphs are acceptable for GraKeL)
        K = self.wl_kernel.transform(graphs_grakel)
        print("WL Kernel matrix shape:", K.shape)
        return K

    def fit_predict(self, dataset):
        K, data_labels = self.__fit_kernel__(dataset)

        embeddings = self.kpca.fit_transform(K)
        labels_kpca = self.kmeans.fit_predict(embeddings)
        #
        # visualize_clusters(embeddings, labels_kpca, data_labels)

        return embeddings, labels_kpca

    def predict(self, dataset):
        K = self.__transform_kernel__(dataset)

        embeddings = self.kpca.transform(K)
        labels_kpca = self.kmeans.predict(embeddings)

        # visualize_clusters(embeddings, labels_kpca, data_labels)

        return embeddings, labels_kpca


def cluster_dataset_spectral(K, n_clusters=3):
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(K)
    print("Cluster labels (Spectral Clustering):", labels)


def cluster_dataset_kpca_kmeans(K, n_clusters=5):
    # Use Kernel PCA to project the data into a Euclidean space
    kpca = KernelPCA(n_components=10, kernel="precomputed", random_state=42)
    embeddings = kpca.fit_transform(K)

    # Now perform k-means clustering on the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels_kpca = kmeans.fit_predict(embeddings)
    print("Cluster labels (Kernel PCA + k-means):", labels_kpca)
    return embeddings, labels_kpca


def visualize_clusters(embeddings, cluster_labels, data_labels):
    """
    Visualizes the embeddings using t-SNE.

    Two subplots are created:
      - Left: Colored by cluster labels (e.g. obtained from spectral clustering)
      - Right: Colored by original data labels (if available, e.g. ground-truth labels)

    Parameters:
    -----------
    embeddings : array-like of shape (n_samples, n_features)
        The high-dimensional embeddings that will be reduced to 2D.

    cluster_labels : array-like of shape (n_samples,)
        The cluster assignments for each embedding.

    data_labels : array-like of shape (n_samples,)
        The ground-truth labels (or any other set of labels you want to compare against).
    """
    # Use t-SNE to reduce dimensions to 2
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Create two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left subplot: Clusters obtained from clustering algorithm
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=cluster_labels, cmap='viridis', s=50)
    axes[0].set_title("Clusters (t-SNE Visualization)")
    axes[0].set_xlabel("t-SNE Dimension 1")
    axes[0].set_ylabel("t-SNE Dimension 2")
    fig.colorbar(scatter1, ax=axes[0], label="Cluster Label")

    # Right subplot: Ground-truth or provided data labels
    scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                               c=data_labels, cmap='viridis', s=50)
    axes[1].set_title("Data Labels (t-SNE Visualization)")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")
    fig.colorbar(scatter2, ax=axes[1], label="Data Label")

    plt.tight_layout()
    plt.show()


def compute_wl_loss(embeddings, data_list):
    wl_measures = compute_wl_measures(data_list)

    dists = torch.cdist(embeddings)

    errs = (wl_measures - dists)

    return errs * errs