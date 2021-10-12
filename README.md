# AS-GCN
This repository contains the demo code of the paper:
>[AS-GCN: Adaptive Semantic Architecture of GraphConvolutional Networks for Text-Rich Networks]

which has been accepted by *ICDM2021*.
## Dependencies
* Python3
* NumPy
* SciPy
* scikit-learn
* NetworkX
* DGL
* PyTorch
## Example
* `python train.py --dataset_str "hep_small" --dataset "data/word_data/hep_small/hep_small.pickle.bin" --out_dim 3 --num_head 1 --epoch 2000 --device cuda:0`

Please refer to the code for detailed parameters.
## Acknowledgements
The demo code is implemented based on [BiTe-GCN: A New GCN Architecture via BidirectionalConvolution of Topology and Features on Text-Rich Networks] (https://arxiv.org/pdf/2010.12157.pdf).
## Citing
    @inproceedings{as-gcn,
     title={AS-GCN: Adaptive Semantic Architecture of GraphConvolutional Networks for Text-Rich Networks},
     author={Zhizhi Yu, Di Jin, Ziyang Liu, Dongxiao He, Xiao Wang, Hanghang Tong, and Jiawei Han},
     booktitle = {ICDM},
     year={2021}
    }


