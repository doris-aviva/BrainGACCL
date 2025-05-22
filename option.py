import argparse
import os
import csv


def parse():
    parser = argparse.ArgumentParser(description='BrainGACCL')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest Wcheckpoint (default: none)')

    # seed
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training')
    parser.add_argument('--random_state', type=int, default=42, help='seed for k fold split')

    # dataset
    parser.add_argument('--isLEUVEN', type=bool, default=False, help='whether dataset is LEUVEN')
    parser.add_argument('--isUCLA', type=bool, default=False, help='whether dataset is UCLA')
    parser.add_argument('--isADNI', type=bool, default=False, help='whether dataset is ADNI')
    parser.add_argument('--data_path', type=str, default='./datasets/NYU116.mat', help='path of data file')
    parser.add_argument('--sparsity', type=int, default=30, help='degree of sparsity')
    parser.add_argument('--self_loop', type=bool, default=True,
                        help='whether to include self-loops when computing the sparsity threshold')
    parser.add_argument('--num_graphs', type=int, default=184, help='graph number')
    parser.add_argument('--num_nodes', type=int, default=116, help='node number')
    parser.add_argument('--k_fold', type=int, default=5, help='the fold number')
    parser.add_argument('--minibatch_size', type=int, default=32, help='batch size')

    # model
    parser.add_argument('--num_dataset_features', type=int, default=116, help='num of node feature')

    parser.add_argument('--input_dim', type=int, default=116, help='input dimension of the model')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of the model')
    parser.add_argument('--num_classes', type=int, default=2, help='class number')
    parser.add_argument('--type', type=str, default='mean', choices=['mean', 'sum', 'concatenate'],
                        help='readout function')

    # attribute augmentor
    parser.add_argument('--hidden_dim_attr_aug', type=int, default=128, help='hidden dimension of the attribute augmentor')
    parser.add_argument('--encoder', type=str, default='GIN_NodeWeightEncoder', help='encoder of the attribute augmentor')

    # structure augmentor
    parser.add_argument('--hidden_dim_stru_aug', type=int, default=64, help='hidden dimension of the structure augmentor')
    parser.add_argument('--drop_ratio', type=float, default=0.0, help='drop_ratio')
    parser.add_argument('--pooling_type', type=str, default='standard', help='pooling type')
    parser.add_argument('--mlp_dim', type=int, default=64, help='dim of mlp')

    # train
    parser.add_argument('--num_epochs', type=int, default=50, help='epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')
    parser.add_argument('--lr_model', type=float, default=0.0008, help='learning rate')
    parser.add_argument('--lr_view', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay_view', type=float, default=1e-3, help='weight_decay')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='lr_decay_factor')
    parser.add_argument('--lr_decay_step_size', type=int, default=30, help='lr_decay_step_size')

    # path
    parser.add_argument('--save_dir', type=str, default='./results', help='folder of saved results')

    # checkpoint
    parser.add_argument('--filename', type=str, default='checkpoint1000_{:03d}_nyu.pth.tar', help='filename of the checkpoint')

    # additional configs:
    parser.add_argument('--pretrained', type=str, default='checkpoint1000_049_nyu.pth.tar', help='path to load pretrained checkpoint')

    # get parameters
    argv = parser.parse_args()

    return argv

