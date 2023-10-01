import torch
import torch.nn.functional as F
import path, os
from utils.utils import *
from models import *
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import config
# 配置参数
args = config.config()
# 按照PCI_BUS_ID顺序从0开始排列GPU设备
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# 配置显卡上的芯片GPU/CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# seed()用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed()值，则每次生成的随即数都相同；不同的随机种子能够生成不同的随机数；
seed = args.seed
torch.manual_seed(seed)

# initialize parameters
dataroot = path.Path(args.dataroot).expanduser()
datadir = dataroot / f'{args.dataname}.mat'
model_name = args.model_name
k_nearest = 10

# 加载data
Xs, y, mask_train, mask_val  = load_data(datadir, selected_mod=(0, 1, 2, 3, 4))

split_ratio = args.split_ratio
if split_ratio:
    mask_train, mask_val = get_split(y, 1. / split_ratio)

ytrs = torch.index_select(y, 0, torch.LongTensor(mask_train))

if args.balanced:
    mask_train, mask_val = get_balanced_sub_idx(ytrs,mask_val)
    idx_val = np.arange(len(mask_val))
    idx_tr = np.arange(len(mask_train))

    Xtrs = [(torch.index_select(Xs[imod],0,torch.LongTensor(mask_train))) for imod in (0, 1, 2, 3, 4)]
    Xvals = [(torch.index_select(Xs[imod],0,torch.LongTensor(mask_val))) for imod in (0, 1, 2, 3, 4)]

    ytrs = torch.index_select(y,0,torch.LongTensor(mask_train))
    yvals = torch.index_select(y,0,torch.LongTensor(mask_val))
else:
    idx_val = np.arange(len(mask_val))
    idx_tr = np.arange(len(mask_train))

    Xtrs = [(torch.index_select(Xs[imod], 0, torch.LongTensor(mask_train))) for imod in (0, 1, 2, 3, 4)]
    Xvals = [(torch.index_select(Xs[imod], 0, torch.LongTensor(mask_val))) for imod in (0, 1, 2, 3, 4)]

    ytrs = torch.index_select(y, 0, torch.LongTensor(mask_train))
    yvals = torch.index_select(y, 0, torch.LongTensor(mask_val))


# init H and X
tmpDataDir = path.Path(f'data/{args.dataname}')
tmpDataDir.makedirs_p()

if args.balanced:
    Htrs = [
        generate_H(Xtrs[imod], k_nearest, tmpDataDir / f'transductive_tr{imod}_k={k_nearest}_balanced.pt')
        for imod in ((0, 1, 2, 3, 4))
    ]
    Hvals = [
        generate_H(Xvals[imod], k_nearest, tmpDataDir / f'transductive_val{imod}_k={k_nearest}_balanced.pt')
        for imod in ((0, 1, 2, 3, 4))
    ]
else:
    Htrs = [
        generate_H(Xtrs[imod], k_nearest, tmpDataDir / f'transductive_tr{imod}_k={k_nearest}.pt')
        for imod in ((0, 1, 2, 3, 4))
    ]
    Hvals = [
        generate_H(Xvals[imod], k_nearest, tmpDataDir / f'transductive_val{imod}_k={k_nearest}.pt')
        for imod in ((0, 1, 2, 3, 4))
    ]

# Hs = create_sparse_H(hyedge_concat(Hs)).to(device)
Htrs = create_sparse_H(hyedge_concat(Htrs)).to(device)
Hvals = create_sparse_H(hyedge_concat(Hvals)).to(device)

Xs = torch.hstack(Xs).to(device)
Xtrs = torch.hstack(Xtrs).to(device)
Xvals = torch.hstack(Xvals).to(device)

ntrfeats = Xtrs.shape[1]
nvalfeats = Xvals.shape[1]

y,ytrs,yvals, mask_train, mask_val ,idx_val,idx_tr= y.to(device),ytrs.to(device),yvals.to(device), torch.Tensor(mask_train).long().to(device),torch.Tensor(mask_val).long().to(device),torch.Tensor(idx_val).long().to(device),torch.Tensor(idx_tr).long().to(device)

nclass = y.max().item() + 1

nlayer, nhid, dropout = args.nlayer, args.nhid, args.dropout


Models = {
    'HGNN': HGNN,
    'ResHGNN': ResHGNN,

}

modeltr = Models[model_name](args, ntrfeats, nhid, nclass, nlayer, dropout).to(device)

optimizertr = torch.optim.Adam(modeltr.parameters(), lr=0.001)

schedulartr = torch.optim.lr_scheduler.MultiStepLR(optimizertr, milestones=[100], gamma=0.9)


#### config the output directory
dataname = args.dataname
out_dir = path.Path( f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}_{split_ratio}/{seed}' )

import shutil 
if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

from logger import get_logger
baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
baselogger.info(args)


def train():
    modeltr.train()
    optimizertr.zero_grad()
    if args.dataname == 'D4_five_methods'or'D3_five_methods'or'D2_five_methods'or'D1_five_methods':
        pred = modeltr(Xtrs, Htrs,device)

    loss = F.nll_loss(pred[idx_tr], ytrs[idx_tr])
    loss.backward()
    optimizertr.step()
    if schedulartr: schedulartr.step()

    _train_acc = accuracy(pred[idx_tr], ytrs[idx_tr])  # 2023.3

    return _train_acc,loss  # 2023.3


def val():
    modeltr.eval()

    if args.dataname == 'D4_five_methods'or'D3_five_methods'or'D2_five_methods'or'D1_five_methods': # 2023.3
        pred = modeltr(Xvals, Hvals,device) # 2023.3

    _val_acc = accuracy(pred[idx_val], yvals[idx_val]) # 2023.3

    # 2023.10.1
    accuracies = []
    for class_label in range(nclass):
        class_acc = class_accuracy(pred[idx_val], yvals[idx_val], class_label)
        accuracies.append(class_acc)
    accuracies_tensor = torch.tensor(accuracies)

    # 计算每个类别的 F1-score 和整体的 F1-score 2023.10.1
    f1_scores_tensor, macro_f1 = calculate_f1_score(pred[idx_val], yvals[idx_val])
    # 计算每个类别的 MCC 和整体的 MCC 2023.10.1
    overall_mcc, class_mccs_tensor = calculate_mcc(pred[idx_val], yvals[idx_val])
    return _val_acc, accuracies_tensor, macro_f1, f1_scores_tensor, overall_mcc, class_mccs_tensor  # 2023.10.1


best_acc, badcounter = 0.0, 0
best_accuracies = torch.tensor([])# 2023.10
best_macro_f1 = 0.0 # 2023.10.1
best_f1_scores = torch.tensor([])# 2023.10.1
best_overall_mcc = 0.0 # 2023.10.1
best_class_mccs = torch.tensor([])# 2023.10.1

for epoch in range(1, args.epochs+1):
    train_acc, loss = train()  # 2023.3
    val_acc, accuracies, macro_f1, f1_scores, overall_mcc, class_mccs = val() # 2023.10.1
    if val_acc > best_acc:
        best_acc = val_acc
        best_accuracies = accuracies  # 2023.10
        badcounter = 0
    else:
        badcounter += 1
    #  2023.10.1
    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_f1_scores = f1_scores
    #  2023.10.1
    if overall_mcc > best_overall_mcc:
        best_overall_mcc = overall_mcc
        best_class_mccs = class_mccs

    if badcounter > args.patience: break
    # 2023.10.1
    best_accuracies_str = ', '.join([f'{acc:.5f}' for acc in best_accuracies.tolist()])
    best_f1_scores_str = ', '.join([f'{f1:.5f}' for f1 in best_f1_scores.tolist()])
    best_class_mccs_str = ', '.join([f'{mcc:.5f}' for mcc in best_class_mccs.tolist()])
    baselogger.info(
        f'Epoch: {epoch}, Loss: {loss:.4f}, Train:{train_acc:.3f}, Val:{val_acc:.3f}, Best Val acc:{best_acc:.5f}, Best accs:[{best_accuracies_str}, Best macro f1:{best_macro_f1:.5f}, Best f1s:[{best_f1_scores_str}, best overall mcc:{best_overall_mcc:.5f}, Best mccs:[{best_class_mccs_str}]')
