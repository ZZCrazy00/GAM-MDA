import torch
import argparse
from utils import get_data, MaskPath, print_result, set_seed
from model import GNNEncoder, EdgeDecoder, GAM
import matplotlib.pyplot as plt
# main parameter
parser = argparse.ArgumentParser()
parser.add_argument('--layer', default="gcn", help="sage, gcn, gin, gat, gat2")
parser.add_argument('--seed', type=int, default=2023, help="Random seed for model and dataset.")
parser.add_argument('--num_encoder', type=int, default=2, help="numbers of GNN encoder")
parser.add_argument('--num_decoder', type=int, default=2, help="numbers of Edge decoder")
parser.add_argument('--walk_length', type=int, default=3, help="length of walk")
parser.add_argument('--p', type=float, default=0.3, help='Mask ratio')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate in optimizer')
parser.add_argument('--wd', type=float, default=5e-4, help='weight decay in optimizer')
parser.add_argument('--times', type=int, default=10, help="numbers of training times")
parser.add_argument('--epoch', type=int, default=1000, help="numbers of training epoch")
args = parser.parse_args()
set_seed(args.seed)

data = get_data()
mask = MaskPath(p=args.p, num_nodes=len(data['train'].x), walk_length=args.walk_length)
encoder = GNNEncoder(len(data['train'].x[0]), 128, 256, num_layers=args.num_encoder, layer=args.layer)
edge_decoder = EdgeDecoder(256, 64, 1, num_layers=args.num_decoder)

model = GAM(encoder, edge_decoder, mask).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

all_result = []
for x in range(args.times):
    for epoch in range(args.epoch):
        model.train()
        loss = model.train_epoch(data['train'], optimizer)
    model.eval()
    test_data = data['test']
    z = model.encoder(test_data.x, test_data.edge_index)
    result = model.test(z, test_data.pos_edge_label_index, test_data.neg_edge_label_index, args.layer)
    all_result.append(result)
print_result(all_result)
# plt.show()
