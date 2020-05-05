import os
import argparse

import random
import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from data import Data
from models import DistMult, CP, CPh, SimplE, ComplEx, Quaternion, TransE
from experiments import Experiment


def get_config():
    """
    Store all configurations with default values.
    Parse values from command line.
    Can be updated from code.
    """
    # ===============
    # Define default values
    # ===============
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--model', default='', type=str, help='the class name of the model for experiment')
    parser.add_argument('--seed', default=7, type=int, help='for replication, used with random.seed(seed), np.random.seed(seed), and tf.set_random_seed(seed)')
    parser.add_argument('--debug', default=0, type=int, help='debug mode, print information for debug: 0, 1')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--work_threads', default=1, type=int, help='number of cpu threads for data processing, to be implemented')
    parser.add_argument('--check_period', default=5, type=int, help='how many epochs after which to check results on train/valid/test')
    parser.add_argument('--early_stop', default=0, type=int, help='to early stop by mrr on valid or not: 0, 1')
    parser.add_argument('--patience', default=10, type=int, help='how many times for patience before early stop')

    # Data and log
    parser.add_argument('--in_path', default='../datasets/wn18/', type=str, help='general data dir: ../datasets/wn18/, ../datasets/fb15k/, ../datasets/wn18rr/, ../datasets/fb15k-237/, ../datasets/KG30C/, ../datasets/KG94C/, ../datasets/KG20C/full_cs/, ../datasets/KG20C/dropy_cs/, ../datasets/KG20C/dropydd_cs/')
    parser.add_argument('--out_path', default='', type=str, help='general output dir: "" (do not write), ../result/temp, ../result/mei')
    parser.add_argument('--model_file_name', default='', type=str, help='default model file name to save')
    parser.add_argument('--log', default=1, type=int, help='to print training progress or not: 1, 0')
    parser.add_argument('--export_steps', default=0, type=int, help='how many epoch for auto save: 0 means not auto saving')
    parser.add_argument('--tb_logdir', default='../result/tb/', type=str, help='tensorboard log dir')

    # Embedding
    parser.add_argument('--D', default=1, type=int, help='multi-embedding size')
    parser.add_argument('--Ce', default=200, type=int, help='ent emb components size: 1 for DistMult, 2 for CP/CPh/SimplE/ComplEx, 4 for Quaternion')
    parser.add_argument('--Cr', default=200, type=int, help='rel emb components size: 1 for DistMult, 2 for CP/CPh/SimplE/ComplEx, 4 for Quaternion')

    # Optimization
    parser.add_argument('--sampling', default='negsamp', type=str, help='which negative sampling method: negsamp (negative sampling)')

    parser.add_argument('--batch_size', default=128, type=int, help='mini batch size')
    parser.add_argument('--batch_size_test', default=0, type=int, help='batch size test, optional: 0 (full batch)')
    parser.add_argument('--neg_ratio', default=5, type=int, help='int, average negative ratio for each triple, note that head and tail are both sampled randomly')
    parser.add_argument('--max_epoch', default=500, type=int, help='max epochs')

    parser.add_argument('--opt_method', default='adam', type=str, help='optimizer: adam, sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1, type=float, help='learning rate decay exponentially: lr_decayed = lr * lr_decay**epoch, 1 (no decay)')

    parser.add_argument('--loss_mode', default='cross-entropy', type=str, help='cross-entropy, softplus, softmax-cross-entropy')

    parser.add_argument('--margin', default=.5, type=float, help='margin for margin loss in transx model: .5')
    parser.add_argument('--binary_weight', default=.5, type=float, help='weight of binary cross-entropy loss in mix loss: .5')

    # Regularization
    parser.add_argument('--lmbda', default=.0, type=float, help='l2 reg in general')
    parser.add_argument('--lmbda_ent', default=.0, type=float, help='l2 reg for entity emb')
    parser.add_argument('--lmbda_rel', default=.0, type=float, help='l2 reg for relation emb')
    parser.add_argument('--lmbda_params', default=.0, type=float, help='l2 reg for other params')
    parser.add_argument('--reg_n3', default=0, type=int, help='N3 reg or not: 0, 1')

    parser.add_argument('--constraint', default='', type=str, help='constraint input entity emb: "", unitnorm, nonneg...')
    parser.add_argument('--to_constrain', default='', type=str, help='which emb to constrain: "", colent, colrel, rowent, rowrel, fullent, fullrel')

    parser.add_argument('--droprate', default=.0, type=float, help='droprate for dropout in general')
    parser.add_argument('--droprate1', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate2', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate3', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate4', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate5', default=.0, type=float, help='droprate for dropout in MEI')
    parser.add_argument('--droprate6', default=.0, type=float, help='droprate for dropout in MEI')

    parser.add_argument('--bn', default=0, type=int, help='batchnorm or not in general: 0, 1')
    parser.add_argument('--bn1', default=0, type=int, help='batchnorm or not in MEI: 0, 1')
    parser.add_argument('--bn2', default=0, type=int, help='batchnorm or not in MEI: 0, 1')
    parser.add_argument('--bn3', default=0, type=int, help='batchnorm or not in MEI: 0, 1')
    parser.add_argument('--bn4', default=0, type=int, help='batchnorm or not in MEI: 0, 1')
    parser.add_argument('--bn5', default=0, type=int, help='batchnorm or not in MEI: 0, 1')
    parser.add_argument('--bn6', default=0, type=int, help='batchnorm or not in MEI: 0, 1')

    parser.add_argument('--bn_momentum', default=.99, type=float, help='momentum in batchnorm: .99')
    parser.add_argument('--bn_epsilon', default=1e-3, type=float, help='epsilon in batchnorm: 1e-3')

    # General
    parser.add_argument('--epsilon', default=1e-12, type=float, help='for numerical stable computation')
    parser.add_argument('--max_value', default=1e100, type=float, help='for numerical stable computation')

    config = parser.parse_args()

    # ===============
    # Update values, add more values
    # ===============
    if 'colent' in config.to_constrain:
        config.constrain_axis_ent = 1
    elif 'rowent' in config.to_constrain:
        config.constrain_axis_ent = 2
    elif 'fullent' in config.to_constrain:
        config.constrain_axis_ent = [1, 2]
    else:
        config.constrain_axis_ent = [1, 2]
    if 'colrel' in config.to_constrain:
        config.constrain_axis_rel = 1
    elif 'rowrel' in config.to_constrain:
        config.constrain_axis_rel = 2
    elif 'fullrel' in config.to_constrain:
        config.constrain_axis_rel = [1, 2]
    else:
        config.constrain_axis_rel = [1, 2]

    if not config.model_file_name:
        config.model_file_name = '--'.join('%s=%s' % (str(x), str(vars(config)[x])) for x in (list(vars(config).keys())))  # model file name to save

    model_class_map = {  # the class name of the model for experiment
        'DistMult': DistMult,
        'CP': CP,
        'CPh': CPh,
        'SimplE': SimplE,
        'ComplEx': ComplEx,
        'Quaternion': Quaternion,
        'TransE': TransE}
    config.model_class = model_class_map[config.model]

    # GPU and tensorflow config
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
    config.tfcfg = tf.ConfigProto()
    config.tfcfg.gpu_options.allow_growth = True
    # for gpu in tf.config.experimental.list_physical_devices('GPU'):
    #     tf.config.experimental.set_memory_growth(gpu, True)

    # Use random seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    tf.set_random_seed(config.seed)
    # tf.random.set_seed(config.seed)

    # create necessary output dirs
    if config.out_path:
        os.makedirs(config.out_path, exist_ok=True)
    if config.tb_logdir:
        os.makedirs(config.tb_logdir, exist_ok=True)

    return config


def main():
    # ===============
    # 1. config
    # ===============
    print('Loading config')
    config = get_config()
    print(config.model_file_name)

    # ===============
    # 2. data
    # ===============
    print('Reading data from %s' % config.in_path)
    data = Data(config=config)

    # ===============
    # 3. experiment
    # ===============
    exp = Experiment(data=data, config=config)
    print('Start training')
    exp.run(check_period=config.check_period, early_stop=config.early_stop, patience=config.patience)


    # ===============
    # 4. test
    # ===============
    if 'wn18' in config.in_path:  # Sanity test on wn18
        exp.show_link_prediction(h='06845599', t='03754979', r='_member_of_domain_usage', raw=True)
    if 'fb15k' in config.in_path:  # Sanity test on fb15k
        exp.show_link_prediction(h='/m/08966', t='/m/05lf_', r='/travel/travel_destination/climate./travel/travel_destination_monthly_climate/month', raw=True)
    if 'KG30C' in config.in_path:  # Sanity test
        exp.show_link_prediction(h='7F74B998', t='019EC1A3', r='paper_in_domain', raw=True)
    if 'KG94C' in config.in_path:  # Sanity test
        exp.show_link_prediction(h='7E52972F', t='80D75AD7', r='author_write_paper', raw=True)


if __name__ == '__main__':
    main()
