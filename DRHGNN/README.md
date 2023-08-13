# ResHGNN

This repository contains the source code for the paper *Identification of Membrane Protein Types via deep residual hypergraph neural network*, 





## Citation






## Getting Started

### Prerequisites

Our code requires Python>=3.6. 

We recommend using virtual environment and  install the newest versions of  [Pytorch](https://pytorch.org/).


You also need these additional packages:

* scipy
* numpy
* path



### Datasets

4 membrane protein datasets are available on request from the corresponding author.

 

## Membrane protein Classification Task

We implement the `HGNN`,  `ResHGNN`. You can change the `$model` and the  `$layer`.



```sh
python train.py --dataroot=$DATA_ROOT --dataname=D1_five_methods  --seed=1  --model-name=$model --nlayer=$layer; 
```




## Stability Analysis

Change the split-ratio as you like.

```sh
python train.py --dataroot=$DATA_ROOT --dataname=D1_five_methods   --model-name=$model --nlayer=$layer --split-ratio=2; 
```

## Usage

```
usage: ResHGNN [-h] [--dataroot DATAROOT] [--dataname DATANAME]
                    [--model-name MODEL_NAME] [--nlayer NLAYER] [--nhid NHID]
                    [--dropout DROPOUT] [--epochs EPOCHS]
                    [--patience PATIENCE] [--gpu GPU] [--seed SEED]
                    [--nostdout] [--balanced] [--split-ratio SPLIT_RATIO]
                    [--out-dir OUT_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   the directary of your .mat data (default:
                        ~/data/)
  --dataname DATANAME   data name (D1_five_methods/...) 
  --model-name MODEL_NAME
                        (HGNN, ResHGNN)
  --nlayer NLAYER       number of hidden layers (default: 4)
  --nhid NHID           number of hidden features (default: 128)
  --dropout DROPOUT     dropout probability (default: 0.5)
  --epochs EPOCHS       number of epochs to train (default: 2000)
  --patience PATIENCE   early stop after specific epochs (default: 200)
  --gpu GPU             gpu number to use (default: 0)
  --seed SEED           seed for randomness (default: 1)
  --nostdout            do not output logging info to terminal (default:
                        False)
  --balanced            only use the balanced subset of training labels
                        (default: False)
  --split-ratio SPLIT_RATIO
                        if set unzero, this is for Task: Stability Analysis,
                        new total/train ratio (default: 0)
```

## License

## reference

We do modification and innovation referring to the code of  *Residual Enhanced Multi-Hypergraph Neural Network*  

```
# @inproceedings{icip21-ResMHGNN,
#   title     = {Residual Enhanced Multi-Hypergraph Neural Network},
#   author    = {Huang, Jing and Huang, Xiaolin and Yang, Jie},
#   booktitle = {International Conference on Image Processing, {ICIP-21}},
#   year      = {2021}
# }
```




