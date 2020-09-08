# Shoestring: Graph-Based Semi-Supervised Classification with Severely Limited Labeled Data


This repo covers an reference implementation for our CVPR 2020 paper [Shoestring](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lin_Shoestring_Graph-Based_Semi-Supervised_Classification_With_Severely_Limited_Labeled_Data_CVPR_2020_paper.pdf) in PyTorch.

## Table of Contents 

* [Installation](#installation)
* [Run demos](#run-demos)
* [Parameters](#parameters)
* [Acknowledgements](#acknowledgements)

## Installation

#### 1. Install Anaconda (<https://www.anaconda.com/download/>)

#### 2. Create a new environment and install tensorflow.

Create a new environment with python=3.7. 
```bash
conda create --name NAME_OF_YOUR_ENVIRONMENT python=3.7.3
```

Activate environment.
```bash
conda activate NAME_OF_YOUR_ENVIRONMENT
```

If you have a Cuda-enabled GPU (check <https://developer.nvidia.com/cuda-gpus> for detail), install tensorflow GPU:
```bash
conda install -c anaconda tensorflow-gpu=1.13.1
```
If not, install tensorflow:
```bash
conda install -c conda-forge tensorflow=1.13.1
```

#### 3. Install other packages

```bash
conda install -c anaconda networkx=2.3
conda install -c anaconda scikit-learn=0.21.1
conda install -c conda-forge texttable
```

## Run Demos

#### 2. Run code with parameters
To reproduce the results in our paper

```bash
# Cora
python train.py --pset config_citation.one_label_set --dataset cora --method l1 l2 cos
python train.py --pset config_citation.two_label_set --dataset cora --method l1 l2 cos
python train.py --pset config_citation.five_label_set --dataset cora --method l1 l2 cos
# Citeseer
python train.py --pset config_citation.one_label_set --dataset citeseer --method l1 l2 cos
python train.py --pset config_citation.two_label_set --dataset citeseer --method l1 l2 cos
python train.py --pset config_citation.five_label_set --dataset citeseer --method l1 l2 cos
# Pubmed
python train.py --pset config_citation.one_label_set --dataset pubmed --method l1 l2 cos
python train.py --pset config_citation.two_label_set --dataset pubmed --method l1 l2 cos
python train.py --pset config_citation.five_label_set --dataset pubmed --method l1 l2 cos
# Large Cora
python train.py --pset config_citation.one_label_set --dataset large_cora --method l1 l2 cos
python train.py --pset config_citation.two_label_set --dataset large_cora --method l1 l2 cos
python train.py --pset config_citation.five_label_set --dataset large_cora --method l1 l2 cos
```

## Parameters

- `k` Select the top k probs for each class as the unlabeled data. Between 1 and 0. Default is 0.
- `lam` Weight for similarity calculated using distance. Default is 0.01.
- `pset` Train size and parameters. Options are: config_citation.one_label_set, config_citation.two_label_set, config_citation.five_label_set. Default is config_citation.one_label_set.
- `dataset` Dataset to train. Options are: cora, large_cora, citeseer, pubmed. Default is cora.
- `method` Method to calculate the distance. Options are l1, l2, cos. Default is cos.

## Cite
Please cite our paper if you use this code in your own work:
```
@inproceedings{Lin_2020_CVPR,
	author = {Lin, Wanyu and Gao, Zhaolin and Li, Baochun},
	title = {Shoestring: Graph-Based Semi-Supervised Classification With Severely Limited Labeled Data},
	booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	month = {June},
	year = {2020}
}
```
## Acknowledgements
Thanks for [Kipf's implementation of GCN](https://github.com/tkipf/gcn/) and [Li's implementation of GLP and IGCN](https://github.com/liqimai/Efficient-SSL), on which this repository is initially based.
