import argparse
import time

import torch.nn.functional as F
from torch.nn import Linear

from datasets import *
# from torch_geometric.nn.conv import MessagePassing, GCNConv
from gcn_convNet import OurGCN2ConvNewData
# from torch_geometric.nn import GCN2Conv
from utils import *

# data.adj_t = gcn_norm(data.adj_t)  # Pre-process GCN normalization.

parser = argparse.ArgumentParser()

parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=int, default=5)
parser.add_argument('--hidden', type=int, default=2)
parser.add_argument('--dropout', type=int, default=2)
parser.add_argument('--normalize_features', type=bool, default=False)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--laynumber', type=int, default=64)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--alpha', type=float, default=1)
parser.add_argument('--H0alpha', type=int, default=5)
parser.add_argument('--theta', type=float, default=0.0)
parser.add_argument('--concat', type=str, default='concat')  #

parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--dataset', type=str,
                    default='wisconsin')  # squirrel film chameleon     cornell   texas   wisconsin# cora PubMed CiteSeer citeseer
parser.add_argument('--eachLayer', type=int, default=1)  # 1 is 1 layer, XX otherwise
parser.add_argument('--trainType', type=str, default='val')  # train / val
parser.add_argument('--gamma', type=str, default='parameter')  # metaNet  parameter / XXX
parser.add_argument('--taskType', type=str, default='full')  # full,semi
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--entropy', type=int, default=1)
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
print('traintype is %s, concat is %s' % (args.trainType, args.concat))
print('dataset: %s' % args.dataset)

# config = json.load(open("config/new%s_%s.json"%(args.dataset,args.taskType), 'r'))

# for i in ['full']:
#     for j in ['cora', 'PubMed', 'CiteSeer']:
#         print("config/new%s_%s.json"%(j,i))
#         json.dump(config, open("config/new%s_%s.json"%(j,i), 'w'))

# for i in ['semi']:
#     for j in ['cs', 'physics', 'computers','photo']:
#         print("config/%s_%s.json"%(j,i))
#         json.dump(config, open("config/%s_%s.json"%(j,i), 'w'))

# [args.H0alpha,args.entropy,args.alpha,args.weight_decay, args.hidden, args.dropout] = list(config.values())
print('H0alpha: %s, entropy: %s, alpha: %s, weight_decay: %s, hidden: %s, dropout: %s'
      % (args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout))


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
    def __init__(self, hidden_channels, num_layers, alpha, theta,
                 shared_weights=True, dropout=0.0, gamma='none', eachLayer=True):
        super(Net, self).__init__()

        flag = False
        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(dataset.num_features, hidden_channels))
        if args.concat == 'concat':
            self.lins.append(Linear(hidden_channels * 2, dataset.num_classes))
        else:
            self.lins.append(Linear(hidden_channels, dataset.num_classes))

        self.convs = torch.nn.ModuleList()
        self.convsVal = torch.nn.ModuleList()
        self.dropout = dropout
        self.ParameterList = torch.nn.ParameterList()
        self.retain_score = 0
        # self.pai = torch.nn.Parameter(torch.randn(dataset.data.num_nodes,num_layers))#.to(torch.device('cuda'))
        # self.theta = torch.nn.Parameter(torch.FloatTensor(1))
        # self.theta.data[0] = 0.5
        # self.ParameterList.append(self.pai)
        # self.ParameterList.append(self.theta)
        # self.PaiNet = Linear(hidden_channels, 1, bias=True)
        # self.convsVal.append(self.PaiNet)
        # self.PaiNet2 = Linear(hidden_channels*2, 1, bias=True)
        # self.convsVal.append(self.PaiNet2)

        if args.concat == 'concat':
            self.MetaNet = Linear(hidden_channels * 2, 1)
        else:
            self.MetaNet = Linear(hidden_channels, 1)
        self.convsVal.append(self.MetaNet)

        self.paramsVal = list(self.convsVal.parameters())

        for layer in range(num_layers):
            self.convs.append(
                OurGCN2ConvNewData(hidden_channels, alpha, theta, layer + 1,
                                   shared_weights, normalize=False))

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
        # if args.concat == 'concat':
        #     preds.append(torch.cat((x, x_0), 1))
        # else:
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

            conv.beta = 0.0  # log( args.theta / count + 1) #0.0#torch.log((torch.sigmoid(self.theta) / count + 1)) # #log( 0.5 / count + 1) #
            # print("alpha")
            # print(conv.alpha)
            # print(self.theta)
            edge_index = adj_t._indices()
            edge_weight = adj_t._values()
            x = conv(x, x_0, edge_index, edge_weight)
            x = x.relu()

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
        retain_score = self.MetaNet(pps)  # pps # n*(L+1)*1
        retain_score = retain_score.squeeze()  # n*(L+1)
        # retain_score = torch.sigmoid(retain_score, -1)  # n*(L+1)
        retain_score, ent_loss = get_t(retain_score)
        self.retain_score = retain_score
        retain_score = retain_score.unsqueeze(1)  # n*1*(L+1)

        x = torch.matmul(retain_score, xMatrix).squeeze()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)

        return x.log_softmax(dim=-1), ent_loss


def train(model, optimizer, data, adj):
    model.train()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True
    optimizer.zero_grad()
    if args.trainType != 'train':  # Val not train

        # for param in model.ParameterList:
        #     param.requires_grad = False
        for param in model.paramsVal:
            param.requires_grad = False

    # try:
    out, ent_loss = model(data.x, adj)  # data.edge_index
    # except:
    #     out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask]) - ent_loss
    loss.backward()
    optimizer.step()
    return float(loss)


def Val_train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

    #
    # for param in model.lins:
    #     param.requires_grad = True
    # for param in model.convs.parameters():
    #     param.requires_grad = True
    # for param in model.ParameterList:
    #     param.requires_grad = True
    for param in model.paramsVal:
        param.requires_grad = True
    # for param in model.MetaNet.parameters():
    #     param.requires_grad = True
    out, ent_loss = model(data.x, adj)  # data.edge_index
    loss = F.nll_loss(out[data.val_mask], data.y[data.val_mask]) - ent_loss
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, optimizer, data, device):
    model.eval()
    # try:
    out, _ = model(data.x, adj)  # data.edge_index
    # except:
    #     out = model(data)
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        accs_loss.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
        accs_loss.append(F.nll_loss(out[mask], data.y[mask].to(device)))
    return accs_loss


@torch.no_grad()
def testGroup(model, optimizer, data, adj, dataset, device):
    model.eval()
    out, ent_loss = model(data.x, adj)
    pred, accs_loss = out.argmax(dim=-1), []
    loss = []

    groupL = getDegreeGroup(dataset)
    for group in groupL:
        if len(group) != 0:
            FalseTensor = torch.tensor(dataset.data.train_mask)
            for i in range(dataset.data.num_nodes):
                FalseTensor[i] = False
            groupmask = torch.tensor(FalseTensor).index_fill_(0, torch.tensor(group).to(device), True)
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


def allTrain(data, adj):
    device = torch.device('cuda')
    model = Net(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta,
                shared_weights=True, dropout=args.dropout, gamma=gamma, eachLayer=args.eachLayer).to(device)
    # model = Net1(hidden_channels=args.hidden, num_layers=args.laynumber, alpha=args.alpha, theta=args.theta,
    #             shared_weights=True, dropout=args.dropout, gamma=gamma, eachLayer=args.eachLayer).to(device)

    # model = Net1(dataset, hidden_channels=64, num_layers=8, alpha=0.2, theta=1.5,
    #     shared_weights=True, dropout=0.6).to(device)
    data = data.to(device)

    groupL = getDegreeGroup(dataset)
    # optimizer = torch.optim.Adam([
    #     dict(params=model.convs.parameters(), weight_decay=args.weight_decay),
    #     dict(params=model.lins.parameters(), weight_decay=args.weight_decay),
    #     dict(params=model.ParameterList,weight_decay=args.weight_decay),
    #     #dict(params=model.theta, weight_decay=0.01),
    #     dict(params=model.convsVal.parameters(), weight_decay=args.weight_decay),
    # ], lr=0.01)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=args.weight_decay)
    # convsVal, ParameterList, lins, convs
    t_total = time.time()
    bad_counter = 0
    best = 999999999
    best_epoch = 0
    acc = 0
    best_test = 0
    bigest_test = 0
    bigest_epoch = 0
    patienceEpoch = 0
    patienceTest = 0
    for epoch in range(args.epochs):
        loss_tra = train(model, optimizer, data, adj)
        if args.trainType != 'train':
            _ = Val_train(model, optimizer, data)
        train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test(model, optimizer, data, device)
        accs_loss = testGroup(model, optimizer, data, adj, dataset, device)

        listN = []
        for i in range(int(len(accs_loss) / 2)):
            listN.append(accs_loss[i * 2])
        print(listN)
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

            # torch.save(model.state_dict(), 'params%s.pkl' % args.laynumber)
            # if (epoch + 1) > 50:
            # out, _ = model(data.x, data.edge_index)
            # outputT = (model.retain_score.cpu().detach().numpy())
            # newarray = np.zeros((len(groupL),outputT.shape[1]))
            # for i in range(len(groupL)):
            #     temp = outputT[groupL[i],:]
            #     newarray[i,:] = np.mean(temp, axis=0)
            #
            # # checkpt_file = '%sresultpiTget__' % args.taskType + args.dataset + '_ConcatbestTest_%s' % tmp_test_acc \
            # #                + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
            # #                    args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden,
            # #                    args.dropout) + '.csv'
            # checkpt_file = '%sresultPiTrue%s__' % (args.taskType,args.laynumber) + args.dataset + 'Epoch_%s' % epoch + '.csv'
            # np.savetxt(checkpt_file, newarray, delimiter=',')

        else:
            bad_counter += 1

        if bad_counter == args.patience:
            patienceEpoch = epoch
            patienceTest = tmp_test_acc
            break

    # if args.test:
    #     acc = test()[1]

    print("Train cost: {:.4f}s".format(time.time() - t_total))
    try:
        print('Max Load {}th epoch'.format(bigest_epoch))
        print("Max Test", "acc.:{:.1f}".format(bigest_test * 100))
    except:
        bigest_epoch = 0
        bigest_test = 0
    print('Load {}th epoch'.format(best_epoch))
    print("Test", "acc.:{:.1f}".format(best_test * 100))
    print('patience Load {}th epoch'.format(patienceEpoch))
    print("patience Test", "acc.:{:.1f}".format(patienceTest * 100))
    model.load_state_dict(torch.load('params%s.pkl' % args.laynumber))
    train_acc, train_loss, val_acc, val_loss, tmp_test_acc, test_loss = test(model, optimizer, data, device)
    accs_loss = testGroup(model, optimizer, data, adj, dataset, device)
    import numpy as np
    listN = []
    for i in range(int(len(accs_loss) / 2)):
        listN.append(accs_loss[i * 2])
    print(listN)
    np.savetxt('getTfile%s%s.csv' % (args.laynumber, args.dataset), np.array(listN), delimiter=',')

    return (bigest_epoch, bigest_test, best_epoch, best_test, patienceEpoch, patienceTest, t_total, model)


# if args.dataset == "chameleon" or args.dataset == "cornell" or args.dataset == "texas" or args.dataset == "wisconsin":
from process import *

datastr = args.dataset
acc_list = []
for i in range(10):
    splitstr = 'splits/' + datastr + '_split_0.6_0.2_' + str(i) + '.npz'
    adj, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr, splitstr)
    adj = adj.to(torch.device('cuda:0'))
    dataset = get_planetoid_dataset('cora', "True")
    dataset.data.train_mask = idx_train
    dataset.data.test_mask = idx_test
    dataset.data.val_mask = idx_val
    dataset.data.edge_index = adj._indices()
    # dataset.data.edge_index = torch.LongTensor(np.concatenate([adj.row.reshape(1, -1), adj.col.reshape(1, -1)], axis=0))
    # dataset.data.edge_index = torch.long(np.concatenate([adj.row.reshape(1, -1), adj.col.reshape(1, -1)], axis=0)) #adj._indices()
    dataset.data.x = features
    dataset.data.y = labels
    # dataset.data.edge_attr = adj._values()
    data = dataset.data

    # dataset.num_features = data.num_features
    # dataset.num_classes = labels.max().numpy()+1
    bigest_epoch, bigest_test, best_epoch, best_test, patienceEpoch, patienceTest, t_total, model = allTrain(data, adj)

    acc_list.append(best_test)
    print(i, ": {:.2f}".format(acc_list[-1]))
    print(acc_list)
print("Train cost: {:.4f}s".format(time.time() - t_total))
print(acc_list)
print("Test acc.:{:.2f}".format(np.mean(acc_list)))

# checkpt_file = 'fullresult/%s%sfullresultpiTgetT_' % (args.laynumber,args.taskType) + args.dataset + '_ConcatbestTest_%s' % np.mean(acc_list) + '_H0alpha%s_entropy%s_alpha%s_weight_decay%s_hidden%s_dropout%s' % (
#     args.H0alpha, args.entropy, args.alpha, args.weight_decay, args.hidden, args.dropout) + '.txt'
#
# np.savetxt(checkpt_file,acc_list,delimiter=' ')
