import argparse
import os
import sys
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='cora',
                    choices=['cora', 'pubmed', 'citeseer', 'amazoncom', 'amazonpho',
                             'coauthorcs', 'coauthorph', 'amazon-ratings', 'questions', 'chameleon-filtered',
                             'squirrel-filtered', 'minesweeper', 'roman-empire', 'wiki-cooc', 'penn94',
                             'blogcatalog', 'flickr'], help='dataset')
parser.add_argument('--method', type=str, default='gcn', choices=['gcn', 'appnp', 'gt', 'gat', 'prognn', 'gen',
                                                                  'gaug', 'idgl', 'grcn', 'sgc', 'jknet', 'slaps',
                                                                  'gprgnn', 'nodeformer', 'segsl', 'sublime',
                                                                  'stable', 'cogsl', 'lpa', 'link', 'linkx', 'wsgnn'], help="Select methods")
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=str, default='0', help="Visible GPU")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from opengsl.config import load_conf
from opengsl.data import Dataset
from opengsl import ExpManager
from opengsl.method import *

if args.config is None:
    conf = load_conf(method=args.method, dataset=args.data)
else:
    conf = load_conf(args.config)
conf.analysis['save_graph'] = False
print(conf)

dataset = Dataset(args.data, feat_norm=conf.dataset['feat_norm'], path='data')


method = eval('{}Solver(conf, dataset)'.format(args.method.upper()))
exp = ExpManager(method,  save_path='records')
acc_save, std_save = exp.run(n_runs=10, debug=args.debug)

if not os.path.exists('results'):
    os.makedirs('results')
if os.path.exists('results/performance.csv'):
    records = pd.read_csv('results/performance.csv')
    records.loc[len(records)] = {'method':args.method, 'data':args.data, 'acc':acc_save, 'std':std_save}
    records.to_csv('results/performance.csv', index=False)
else:
    records = pd.DataFrame([[args.method, args.data, acc_save, std_save]], columns=['method', 'data', 'acc', 'std'])
    records.to_csv('results/performance.csv', index=False)