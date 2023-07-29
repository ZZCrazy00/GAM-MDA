import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.metrics import roc_curve, auc, average_precision_score, precision_recall_curve


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def draw_auc(y, pred, l):
    fpr, tpr, _ = roc_curve(y, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{}:AUC = %0.4f'.format(l) % roc_auc)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)


def draw_aupr(y, pred, l):
    average_precision = average_precision_score(y, pred)
    precision, recall, _ = precision_recall_curve(y, pred)
    plt.plot(recall, precision, label='{}:AUPR = %0.4f'.format(l) % average_precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('AUPR Curve')
    plt.legend(loc='lower right')
    plt.grid(True)


def print_result(result):
    metrics = ['auc', 'acc', 'sen', 'pre', 'spe', 'F1', 'mcc']
    metric_values = [[] for _ in range(len(metrics))]
    for i in result:
        for j, val in enumerate(i):
            metric_values[j].append(val)
    metric_values = [np.array(m) for m in metric_values]
    formatted_metrics = []
    for metric, values in zip(metrics, metric_values):
        mean = "{:.4f}".format(values.mean())
        std = "{:.4f}".format(np.std(values))
        formatted_metrics.append(f"{metric}: {mean} ± {std}")
    print(*formatted_metrics)


def mask_path(edge_index, p, walks_per_node, walk_length, num_nodes):
    edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index = sort_edge_index(edge_index, num_nodes=num_nodes)
    row, col = edge_index
    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)
    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])
    n_id, e_id = torch.ops.torch_cluster.random_walk(rowptr, col, start, walk_length, 1.0, 1.0)
    e_id = e_id[e_id != -1].view(-1)
    edge_mask[e_id] = False
    return edge_index[:, edge_mask], edge_index[:, ~edge_mask]


class MaskPath(torch.nn.Module):
    def __init__(self, p, walk_length, num_nodes):
        super(MaskPath, self).__init__()
        self.p = p
        self.walk_length = walk_length
        self.num_nodes = num_nodes

    def forward(self, edge_index):
        remaining_edges, masked_edges = mask_path(edge_index, self.p, 1, self.walk_length, self.num_nodes)
        remaining_edges = to_undirected(remaining_edges)
        return remaining_edges, masked_edges


def calculate_metrics(y_true, y_pred):
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    TN = sum((y_true[i] == 0 and y_pred[i] == 0) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)
    sensitivity = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    specificity = TN / (TN + FP + 1e-10)
    mcc = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    F1_score = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-10)
    return accuracy, sensitivity, precision, specificity, F1_score, mcc


def get_data():
    print("Loading data.")
    miRNA_drug = pd.read_csv("data/data_3000.csv", header=None)
    miRNA_list = list(set(miRNA_drug[0]))  # 701
    drug_list = list(set(miRNA_drug[1]))  # 101
    adj = torch.LongTensor(
        [[miRNA_list.index(x[0]), drug_list.index(x[1]) + len(miRNA_list)] for x in miRNA_drug.values]).T

    miRNA = pd.read_csv("data/miRNA_kmer.csv", header=None)
    drug = pd.read_csv("data/drug_GIN_64.csv", header=None)
    feature = torch.Tensor(miRNA.values.tolist() + drug.values.tolist())

    train_data, _, test_data = T.RandomLinkSplit(num_val=0, num_test=0.2,
                                                 is_undirected=True, split_labels=True,
                                                 add_negative_train_samples=False)(Data(x=feature, edge_index=adj).cuda())
    splits = dict(train=train_data, test=test_data)
    return splits
