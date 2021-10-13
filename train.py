import argparse
import warnings

from src.trainer import *
from src.utils import set_seed
warnings.simplefilter("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # "ntm model"
    parser.add_argument('--topic_num', type=int, default=30)
    parser.add_argument('--check_pt_ntm_model_path', type=str)
    parser.add_argument('--ntm_warm_up_epochs', type=int, default=200)
    parser.add_argument('--model_path', type=str, default="checkpoint_ntm/",
                        help="Path of checkpoints.")
    parser.add_argument('--target_sparsity', type=float, default=0.85,
                        help="Target sparsity for ntm model")
    parser.add_argument('--topic_type', default='z', choices=['z', 'g'], help='use latent variable z or g as topic')
    parser.add_argument('--in_file', default='cleaned.txt')
    parser.add_argument('--learning_rate_ntm', type=float, default=0.0005)
    parser.add_argument('-two_stage', default=False, action='store_true')
    parser.add_argument('-only_train_ntm', default=False, action='store_true')
    parser.add_argument('-load_pretrain_ntm', default=False, action='store_true')
    parser.add_argument('-use_topic_represent', default=True, action='store_true',
                        help="Use topic represent in the topicGCN")


    parser.add_argument('--dataset_str', type=str, default='hep_small')
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--seed', type=int, default=1566911445, help='Random seed.')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--log_interval', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--learning_rate_decay_patience', type=int, default=10)
    parser.add_argument('--learning_rate_decay_factor', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--model', type=str, default='HeteroGAT')
    parser.add_argument('--dataset', type=str, default='data/word_data/hep_small/hep_small.pickle.bin')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--out_dim', type=int, default=3)
    parser.add_argument('--aggr_func', type=str, default='mlp',
                        choices=["mean", "sum", "linear", "mlp", "attention"])
    parser.add_argument('--node_feature', type=str, default='one_hot', choices=['one_hot', 'message_passing','bert','raw'])
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--topwords', type=int, default=10)
    parser.add_argument('--toptopic', type=int, default=2)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--num_head', type=int, default=1)
    parser.add_argument('--residual', type=int, default=0)
    parser.add_argument('--lamuda', type=float, default=0.001)
    args = parser.parse_args()
    args.verbose = bool(args.verbose)
    args.residual = bool(args.residual)
    # setting random seeds
    # set_seed(args.seed, args.device)

    print(args)
    if args.model == "HeteroGAT":
        hetero_gat(args)
    elif args.model == "VanillaGCN":
        vanilla_gcn(args)
    elif args.model == "VanillaGAT":
        vanilla_gat(args)
    else:
        args.verbose = False
        print("################### VanillaGCN ###################")
        vanilla_gcn(args)
        print("################### VanillaGAT ###################")
        vanilla_gat(args)
        print("################### HeteroGAT ###################")
        hetero_gat(args)
