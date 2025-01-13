import argparse
import logging

# from  mcs_pooling.pooling_experiment import run_pooling_experiment
# from mcs_convolution.mcs_conv_experiment import run_mcs_conv_experiment
from hgpsl_pooling.hgpsl_experiment import run_hgp_sl_experiment


def run(run_args):
    # if run_args.exp_name == 'mcs_convolution':
    #     run_mcs_conv_experiment(run_args.dataset_name, run_args.train_num, run_args.val_num, run_args.test_num)
    # elif run_args.exp_name == 'mcs_pooling':
    #     run_pooling_experiment(run_args.dataset_name, run_args.train_num, run_args.val_num, run_args.test_num)
    # elif run_args.exp_name == 'hgpsl_pooling':
    if run_args.exp_name == 'hgpsl_pooling':
        run_hgp_sl_experiment(run_args.dataset_name, run_args.train_num, run_args.val_num, run_args.test_num, run_name=run_args.run_name)
    elif run_args.exp_name == 'knn':
        pass
    else: raise Exception('Wrong Experiment Name')

if __name__ == '__main__':
     
    parser = argparse.ArgumentParser(
        description="This program runs our graph machine learning experiments"
    )
    parser.add_argument("--exp_name", required=True, type=str, default='hgpsl_pooling') # options: knn, mcs_pooling, mcs_convolution
    parser.add_argument("--dataset_name", required=False, type=str, default='PROTEINS')
    parser.add_argument("--train_num", required=False, type=int, default=300)
    parser.add_argument("--val_num", required=False, type=int, default=100)
    parser.add_argument("--test_num", required=False, type=int, default=100)

    parser.add_argument("--workers_num", required=False, type=int, default=4)
    parser.add_argument("--each_class_train", required=False, type=int, default=10)
    parser.add_argument("--each_class_val", required=False, type=int, default=10)
    parser.add_argument("--device", required=False, type=str, default='cpu')
    parser.add_argument("--run_name", required=False, type=str, default=None)

    args = parser.parse_args()

    run(args)

