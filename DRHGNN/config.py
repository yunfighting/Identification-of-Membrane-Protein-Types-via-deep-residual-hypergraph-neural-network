import argparse

def config():
    p = argparse.ArgumentParser("ResHGNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--dataroot', type=str, default=r'.\data', help='the directary of your .mat data')
    p.add_argument('--dataname', type=str, default='D1_five_methods', help='data name (D1_five_methods/D2_five_methods/D3_five_methods/D4_five_methods)')
    p.add_argument('--model-name', type=str, default='ResHGNN', help='(HGNN, ResHGNN)')
    p.add_argument('--nlayer', type=int, default=4, help='number of hidden layers')
    p.add_argument('--nhid', type=int, default=128, help='number of hidden features')
    p.add_argument('--dropout', type=float, default=0.5, help='dropout probability')

    p.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
    p.add_argument('--patience', type=int, default=200, help='early stop after specific epochs')
    p.add_argument('--gpu', type=int, default=0, help='gpu number to use')
    p.add_argument('--seed', type=int, default=1, help='seed for randomness')
    p.add_argument('--nostdout', action="store_true",  help='do not output logging info to terminal')
    p.add_argument('--balanced', action="store_true", default=0, help='only use the balanced subset of training labels')
    p.add_argument('--split-ratio', type=int, default=0,  help='if set unzero, this is for Task: Stability Analysis, new total/train ratio')
    p.add_argument('--out-dir', type=str, default='runs/test',  help='output dir')
    return p.parse_args()