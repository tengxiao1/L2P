import argparse
import time

import torch.nn.functional as F
from torch.nn import Linear

from datasets import *
# from torch_geometric.nn.conv import MessagePassing, GCNConv
from gcn_convNet import OurGCN2Conv
# from torch_geometric.nn import GCN2Conv
from sample import Sampler
from utils import *

# data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

parser = argparse.ArgumentParser()

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=int, default=2)
parser.add_argument('--hidden', type=int, default=0)
parser.add_argument('--dropout', type=int, default=3)
parser.add_argument('--normalize_features', type=bool, default=True)  # False
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--laynumber', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--alpha', type=float, default=0.0)  # 0.1
parser.add_argument('--H0alpha', type=int, default=0)
parser.add_argument('--theta', type=float, default=0.0)
parser.add_argument('--concat', type=str, default='concat')  # concat

parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--dataset', type=str, default='Cora')  # Cora PubMed CiteSeer
parser.add_argument('--eachLayer', type=int, default=1)  # 1 is 1 layer, XX otherwise
parser.add_argument('--trainType', type=str, default='val')  # train / XXX
parser.add_argument('--gamma', type=str, default='parameter')  # metaNet  parameter / XXX
parser.add_argument('--taskType', type=str, default='semi')  # full,semi
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--entropy', type=int, default=4)
parser.add_argument('--gumbleBool', type=bool, default=False)
args = parser.parse_args()
task_type = args.taskType  # full,semi
gamma = args.gamma

import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


set_seed(args.seed)
weightDecayL = [1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
args.weight_decay = weightDecayL[args.weight_decay]
hiddenL = [64, 128, 256]
args.hidden = hiddenL[args.hidden]
args.entropy = args.entropy / 5.0
args.dropout = args.dropout / 5.0
args.H0alpha = args.H0alpha * 0.2
# print('gamma is %s, traintype is %s'%(args.gamma,args.trainType))
print('dataset: %s' % args.dataset)

import json

#try:
 #   config = json.load(open("config/%s_%s.json" % (args.dataset, args.taskType), 'r'))
#except:
 #   config = json.load(open("config/%s_%s.json" % (args.dataset.lower(), args.taskType), 'r'))
# # for i in ['full','semi']:
# #     for j in ['cora', 'PubMed', 'CiteSeer']:
# #         print("config/%s_%s.json"%(j,i))
# #         json.dump(config, open("config/%s_%s.json"%(j,i), 'w'))
[args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout] = list(config.values())
print('H0alpha: %s, entropy: %s, alpha: %s, weight_decay: %s, hidden: %s, dropout: %s'
      % (args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout))

if args.dataset == "Cora" or args.dataset == "CiteSeer" or args.dataset == "PubMed":
    dataset = get_planetoid_dataset(args.dataset, "True")
    # dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    path = dataset.raw_dir  # osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
    sampler = Sampler(args.dataset.lower(), path, task_type)
    # args.dataset.lower()
    # get labels and indexes
    labels, idx_train, idx_val, idx_test = sampler.get_label_and_idxes(True)

    FalseTensor = torch.tensor(dataset.data.train_mask)
    for i in range(dataset.data.num_nodes):
        FalseTensor[i] = False
    if task_type == 'full':
        dataset.data.train_mask = torch.tensor(FalseTensor).index_fill_(0, idx_train.cpu(), True)
        dataset.data.test_mask = torch.tensor(FalseTensor).index_fill_(0, idx_test.cpu(), True)
        dataset.data.val_mask = torch.tensor(FalseTensor).index_fill_(0, idx_val.cpu(), True)
        print('Full-supervised train_mask is %s' % dataset.data.train_mask.numpy().sum())
        print('Full-supervised test_mask is %s' % dataset.data.test_mask.numpy().sum())
        print('Full-supervised val_mask is %s' % dataset.data.val_mask.numpy().sum())

    nfeat = sampler.nfeat
    nclass = sampler.nclass
    # print("nclass: %d\tnfea:%d" % (nclass, nfeat))

    # if args.normalize_features:
    data = dataset[0]
    # else:
    #     data = dataset.data
    # For the mix mode, lables and indexes are in cuda.
    # if args.cuda or args.mixmode:
    #     labels = labels.cuda()
    #     idx_train = idx_train.cuda()
    #     idx_val = idx_val.cuda()
    #     idx_test = idx_test.cuda()

    # print('Previous train_mask is %s' % dataset.data.train_mask.numpy().sum())
    # print('Previous test_mask is %s' % dataset.data.test_mask.numpy().sum())
    # print('Previous val_mask is %s' % dataset.data.val_mask.numpy().sum())
elif args.dataset == "cs" or args.dataset == "physics":
    dataset = get_coauthor_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])

    data_ori = dataset[0]
    from torch_geometric.utils import *
    import networkx as nx

    data_nx = to_networkx(data_ori)
    data_nx = data_nx.to_undirected()
    print("Original #nodes:", data_nx.number_of_nodes())
    data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
    print("#Nodes after lcc:", data_nx.number_of_nodes())
    lcc_mask = list(data_nx.nodes)
    # if args.normalize_features:
    #     data = dataset[0]
    #     data = permute_masks(data, dataset.num_classes, lcc_mask)
    # else:
    data = dataset.data
    data = permute_masks(data, dataset.num_classes, lcc_mask)


elif args.dataset == "computers" or args.dataset == "photo":
    dataset = get_amazon_dataset(args.dataset, args.normalize_features)
    permute_masks = random_coauthor_amazon_splits
    print("Data:", dataset[0])

    data_ori = dataset[0]
    from torch_geometric.utils import *
    import networkx as nx

    data_nx = to_networkx(data_ori)
    data_nx = data_nx.to_undirected()
    print("Original #nodes:", data_nx.number_of_nodes())
    data_nx = data_nx.subgraph(max(nx.connected_components(data_nx), key=len))
    print("#Nodes after lcc:", data_nx.number_of_nodes())
    lcc_mask = list(data_nx.nodes)
    # if args.normalize_features:
    #     data = dataset[0]
    #     data = permute_masks(data, dataset.num_classes, lcc_mask)
    # else:
    data = dataset.data
    data = permute_masks(data, dataset.num_classes, lcc_mask)


# 0.888
# print(data.x.numpy().sum()) #49216
# data=dataset[0] # 0.822
# print(data.x.numpy().sum()) #2708.0012
# orginal
# data=dataset[0]

# data.adj_t = gcn_norm(data.adj_t)

# python GCNII.py --dataset=Cora --trainType val --gamma parameter --taskType semi

def get_t(retain_score):
    tMatrix = 1
    for count in range(retain_score.shape[1]):
        t = 1
        if count == 0:
            tMatrix = torch.sigmoid(retain_score[:, count]).reshape(retain_score.shape[0], 1)
        else:
            if count == retain_score.shape[1] - 1:
                t = 1
            else:
                t = torch.sigmoid(retain_score[:, count])

            for i in range(count):
                t = (1 - torch.sigmoid(retain_score[:, i])) * t

            tMatrix = torch.cat((tMatrix, t.reshape(tMatrix.shape[0], 1)), 1)
    # ent_loss = 0.1 * torch.distributions.Categorical(tMatrix).entropy().mean()
    ent_loss = args.entropy * -(torch.log(tMatrix + 1e-20) * tMatrix).sum(1).mean()
    tMatrix = F.gumbel_softmax(torch.log(tMatrix + 1e-20), tau=1, hard=False)
    # tMatrix = F.gumbel_softmax((tMatrix), tau=1, hard=False)
    return tMatrix, ent_loss


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, alpha, theta, dataset,
                 shared_weights=True, dropout=0.0, gamma='none', eachLayer=True):
        super(Net, self).__init__()

        flag = False
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        # self.lins.append(Linear(hidden_channels, dataset.num_classes))
        if args.concat == 'concat':
            self.lins.append(Linear(hidden_channels * 2, dataset.num_classes))
        else:
            self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        self.convsVal = torch.nn.ModuleList()
        self.dropout = dropout
        self.ParameterList = torch.nn.ParameterList()
        self.retain_score = 0
        self.pai = torch.nn.Parameter(torch.randn(dataset.data.num_nodes, num_layers))  # .to(torch.device('cuda'))
        self.theta = torch.nn.Parameter(torch.FloatTensor(1))
        self.theta.data[0] = 0.5
        self.ParameterList.append(self.pai)
        self.ParameterList.append(self.theta)
        # self.PaiNet = Linear(hidden_channels, 1, bias=True)
        # self.convsVal.append(self.PaiNet)
        # self.PaiNet2 = Linear(hidden_channels*2, 1, bias=True)
        # self.convsVal.append(self.PaiNet2)

        # self.MetaNet = Linear(hidden_channels, 1)
        if args.concat == 'concat':
            self.MetaNet = Linear(hidden_channels * 2, 1)
        else:
            self.MetaNet = Linear(hidden_channels, 1)
        self.convsVal.append(self.MetaNet)
        self.paramsVal = list(self.convsVal.parameters())

        for layer in range(num_layers):
            self.convs.append(
                OurGCN2Conv(hidden_channels, alpha, theta, layer + 1,
                            shared_weights, normalize=True))

    def forward(self, x, adj_t):
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()  # F.dropout(self.lins[0](x).relu(), self.dropout, training=self.training) #
        count = 0
        l = []
        HMatrix = 0
        paiMatrix = 1
        paiList = []
        # self.pai = torch.sigmoid(self.pai)
        preds = []
        xMatrix = []
        # preds.append(x)
        if args.concat == 'concat':
            preds.append(torch.cat((x, x_0), 1))
        else:
            preds.append(x)
        if args.concat == 'concat':
            xMatrix.append(torch.cat((x, args.H0alpha * x_0), 1))
        else:
            xMatrix.append(x)

        for conv in self.convs:
            paiT = 1
            count += 1
            # if count >1:
            x = F.dropout(x, self.dropout, training=self.training)

            # count used in this function

            # if len(self.ParameterList) != 0:
            conv.alpha = 0.0  # args.alpha#0.1

            conv.beta = 0.0  # torch.log((torch.sigmoid(self.theta) / count + 1)) #log( 0.5 / count + 1) #log( 0.5 / count + 1) #
            # print("alpha")
            # print(conv.alpha)
            # print(self.theta)

            x = conv(x, x_0, adj_t)
            x = x.relu()
            # preds.append(x)
            if args.concat == 'concat':
                preds.append(torch.cat((x, x_0), 1))
            else:
                preds.append(x)
            if args.concat == 'concat':
                xMatrix.append(torch.cat((x, args.H0alpha * x_0), 1))
            else:
                xMatrix.append(x)
        xMatrix = torch.stack(xMatrix, dim=1)
        pps = torch.stack(preds, dim=1)  # n*(L+1)*k
        retain_score = self.MetaNet(pps)  # n*(L+1)*1
        retain_score = retain_score.squeeze()  # n*(L+1)
        # retain_score = torch.sigmoid(retain_score, -1)  # n*(L+1)
        m = torch.nn.Softmax(dim=1)

        retain_score = m(retain_score)
        # print(retain_score.min(),retain_score.max())
        ent_loss = args.entropy * -(torch.log(retain_score + 1e-20) * retain_score).sum(1).mean()
        # retain_score,ent_loss = get_t(retain_score)
        self.retain_score = retain_score
        retain_score = retain_score.unsqueeze(1)  # n*1*(L+1)

        x = torch.matmul(retain_score, xMatrix).squeeze()
        # x = torch.matmul(retain_score, xMatrix).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1), 0.0  # ent_loss


device = torch.device('cuda')
model = Net(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta, dataset=dataset,
            shared_weights=True, dropout=args.dropout, gamma=gamma, eachLayer=args.eachLayer).to(device)
data = data.to(device)

optimizer = torch.optim.Adam([
    dict(params=model.convs.parameters(), weight_decay=args.weight_decay),
    dict(params=model.lins.parameters(), weight_decay=args.weight_decay),
    dict(params=model.ParameterList, weight_decay=args.weight_decay),
    # dict(params=model.theta, weight_decay=0.01),
    dict(params=model.convsVal.parameters(), weight_decay=args.weight_decay),
], lr=0.01)


def train():
    model.train()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    optimizer.zero_grad()
    if args.trainType != 'train':  # Val not train
        if gamma == 'parameter':
            for param in model.ParameterList:
                param.requires_grad = False
            for param in model.paramsVal:
                param.requires_grad = False

    out, ent_loss = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) - ent_loss
    loss.backward()
    optimizer.step()
    return float(loss)


def Val_train():
    model.train()
    optimizer.zero_grad()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False
    if gamma == 'parameter':
        for param in model.ParameterList:
            param.requires_grad = True
        for param in model.paramsVal:
            param.requires_grad = True
    out, ent_loss = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]) - ent_loss
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    out, _ = model(data.x, data.edge_index)
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs_loss.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        accs_loss.append(F.nll_loss(out[mask], data.y[mask].to(device)))
    return accs_loss


@torch.no_grad()
def testGroup():
    model.eval()
    out, ent_loss = model(data.x, data.edge_index)
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []

    groupL = getDegreeGroup(dataset)
    for group in groupL:
        FalseTensor = torch.tensor(dataset.data.train_mask)
        for i in range(dataset.data.num_nodes):
            FalseTensor[i] = False
        groupmask = torch.tensor(FalseTensor).index_fill_(0, torch.tensor(group), True)
        # for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs_loss.append(int((pred[groupmask] == data.y[groupmask]).sum()) / int(groupmask.sum()))
        accs_loss.append(F.nll_loss(out[groupmask], data.y[groupmask].to(device)))
    return accs_loss


def getDegreeGroup(dataset):
    edge_index = dataset.data.edge_index
    e0 = edge_index[0]
    dict = {}
    for key in e0:
        dict[int(key)] = dict.get(int(key), 0) + 1

    totalList = 0
    groupL = []
    i = 0
    while totalList < len(dict.keys()):
        group = []
        lower = 2 ** i
        higher = 2 ** (i + 1)
        for key in dict.keys():
            if higher > dict.get(int(key), 0) >= lower:
                group.append(int(key))
        groupL.append(group)
        totalList += len(group)
        # print(group)
        # print(totalList)
        i += 1
    return groupL


groupL = getDegreeGroup(dataset)
t_total = time.time()
bad_counter = 0
best = 999999999
best_epoch = 0
acc = 0
best_test = 0
bigest_test = 0
patienceEpoch = 0
patienceTest = 0
accs_loss = testGroup()
for epoch in range(args.epochs):
    loss_tra = train()
    if args.trainType != 'train':
        _ = Val_train()
    train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test()

    if (epoch + 1) % 1 == 0:
        print('Epoch:{:04d}'.format(epoch + 1),
              'train',
              'loss:{:.3f}'.format(train_loss),
              'acc:{:.2f}'.format(train_acc * 100),
              '| val',
              'loss:{:.3f}'.format(val_loss),
              'acc:{:.2f}'.format(val_acc * 100),
              'acc_test:{:.2f}'.format(tmp_test_acc),
              'best_acc_test:{:.2f}'.format(best_test * 100))
    accs_loss = testGroup()
    listN = []
    for i in range(int(len(accs_loss) / 2)):
        listN.append(accs_loss[i * 2])
    print(listN)
    if val_loss < best:
        best = val_loss
        best_epoch = epoch
        acc = val_acc
        # if best_test <tmp_test_acc:
        if bigest_test < tmp_test_acc:
            bigest_test = tmp_test_acc
            bigest_epoch = epoch
        best_test = tmp_test_acc
        #
        bad_counter = 0
        torch.save(model.state_dict(), 'params%s.pkl' % args.laynumber)

        # if tmp_test_acc > 10.84:
        out, _ = model(data.x, data.edge_index)
        outputT = (model.retain_score.cpu().detach().numpy())
        newarray = np.zeros((len(groupL), outputT.shape[1]))
        for i in range(len(groupL)):
            temp = outputT[groupL[i], :]
            newarray[i, :] = np.mean(temp, axis=0)

            # checkpt_file = '%sresultpiensoft%s__' % (args.taskType,args.laynumber) + args.dataset + '_ConcatbestTest_%s' % tmp_test_acc \
            #                + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
            #                    args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
            #                    args.dropout) + '.csv'
            # checkpt_file = '%sresultpiensoft%s__' % (
            # args.taskType, args.laynumber) + args.dataset + 'Epoch_%s' % epoch + '.csv'
            # np.savetxt(checkpt_file, newarray, delimiter=',')



    else:
        bad_counter += 1

    if bad_counter == args.patience:
        patienceEpoch = epoch
        patienceTest = tmp_test_acc

model.load_state_dict(torch.load('params%s.pkl' % args.laynumber))
train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test()

print('Novel Epoch:{:04d}'.format(epoch + 1),
      'train',
      'loss:{:.3f}'.format(train_loss),
      'acc:{:.2f}'.format(train_acc * 100),
      '| val',
      'loss:{:.3f}'.format(val_loss),
      'acc:{:.2f}'.format(val_acc * 100),
      'acc_test:{:.2f}'.format(tmp_test_acc),
      'best_acc_test:{:.2f}'.format(best_test * 100))

accs_loss = testGroup()
import numpy as np

listN = []
for i in range(int(len(accs_loss) / 2)):
    listN.append(accs_loss[i * 2])
print(listN)
np.savetxt('softfile%s%s.csv' % (args.laynumber, args.dataset), np.array(listN), delimiter=',')

# if args.test:
#     acc = test()[1]

print("Train cost: {:.4f}s".format(time.time() - t_total))
try:
    print('Max Load {}th epoch'.format(bigest_epoch))
    print("Max Test", "acc.:{:.1f}".format(bigest_test * 100))
except:
    1
print('Load {}th epoch'.format(best_epoch))
print("Test", "acc.:{:.1f}".format(best_test * 100))
print('patience Load {}th epoch'.format(patienceEpoch))
print("patience Test", "acc.:{:.1f}".format(patienceTest * 100))
# checkpt_file = '%s%sresultsoft_'% (args.laynumber,args.taskType) + args.dataset + '_ConcatbestTest_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.txt'
#
# np.savetxt(checkpt_file,np.zeros(1),delimiter=' ')
# checkpt_file = '%sresultpiTen_'%args.taskType + args.dataset + '_ConcatbestTest_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
# cudaid = "cuda:" + str(args.dev)
# print(cudaid, checkpt_file)
# torch.save(model.state_dict(), checkpt_file)
# try:
#     if args.concat == 'concat':
#
#
#         if task_type == 'full':
#             if args.dataset == 'cora':
#                 if best_test > 0.86:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.86:
#                     checkpt_file = 'fullresult/' + args.dataset + '/ConcatpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.88:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'CiteSeer':
#                 if best_test > 0.8:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.8:
#                     checkpt_file = 'fullresult/' + args.dataset + '/ConcatpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.8:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'PubMed':
#                 if best_test > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbest_test_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/ConcatpatienceTest_%s' % patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/Concatbigest_test_%s' % bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#         else:
#             if args.dataset == 'cora':
#                 if best_test>0.845:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.845:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.845:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#
#             if args.dataset == 'CiteSeer':
#                 if best_test>0.738:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.738:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.73:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#             if args.dataset == 'PubMed':
#                 if best_test>0.803:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.803:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.805:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#
#             if args.dataset == 'cs':
#                 if best_test>0.9:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.9:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.9:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#             if args.dataset == 'physics':
#                 if best_test>0.94:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.94:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.94:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#             if args.dataset == 'computers':
#                 if best_test>0.84:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.84:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.84:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#
#             if args.dataset == 'photo':
#                 if best_test>0.91:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbest_test_%s'%best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest>0.91:
#                     checkpt_file = 'result/' + args.dataset + '/ConcatpatienceTest_%s'%patienceTest + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test>0.91:
#                     checkpt_file = 'result/' + args.dataset + '/Concatbigest_test_%s'%bigest_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#
#     else:
#         if task_type == 'full':
#             if args.dataset == 'cora':
#                 if best_test > 0.88:
#                     checkpt_file = 'fullresult/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.88:
#                     checkpt_file = 'fullresult/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.88:
#                     checkpt_file = 'fullresult/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'CiteSeer':
#                 if best_test > 0.795:
#                     checkpt_file = 'fullresult/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.795:
#                     checkpt_file = 'fullresult/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.795:
#                     checkpt_file = 'fullresult/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'PubMed':
#                 if best_test > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.9:
#                     checkpt_file = 'fullresult/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#         else:
#             if args.dataset == 'cora':
#                 if best_test > 0.84:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.84:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.85:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'CiteSeer':
#                 if best_test > 0.718:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.718:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.73:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'PubMed':
#                 if best_test > 0.793:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.793:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.805:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'cs':
#                 if best_test > 0.928:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.928:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.928:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'physics':
#                 if best_test > 0.94:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.94:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.94:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'computers':
#                 if best_test > 0.84:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.84:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.84:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#
#             if args.dataset == 'photo':
#                 if best_test > 0.91:
#                     checkpt_file = 'result/' + args.dataset + '/best_test_%s' % best_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif patienceTest > 0.91:
#                     checkpt_file = 'result/' + args.dataset + '/patienceTest_%s' % patienceTest + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
#                 elif bigest_test > 0.91:
#                     checkpt_file = 'result/' + args.dataset + '/bigest_test_%s' % bigest_test + '_normalize%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#                         args.normalize_features, args.entropy, args.alpha, args.weight_decay, args.hidden,
#                         args.dropout) + '.pt'
#                     cudaid = "cuda:" + str(args.dev)
#                     print(cudaid, checkpt_file)
#                     torch.save(model.state_dict(), checkpt_file)
# except:
#     1
#
#
# checkpt_file = '%sresultpiTen_'%args.taskType + args.dataset + '_ConcatbestTest_%s' % best_test + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.pt'
# cudaid = "cuda:" + str(args.dev)
# print(cudaid, checkpt_file)
# torch.save(model.state_dict(), checkpt_file)
