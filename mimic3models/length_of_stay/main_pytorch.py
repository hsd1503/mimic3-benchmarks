from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import argparse
import os
import imp
import re
from tqdm import tqdm

from mimic3models.length_of_stay import utils
from mimic3benchmark.readers import LengthOfStayReader

from mimic3models.preprocessing import Discretizer, Normalizer
from mimic3models import metrics
from mimic3models import keras_utils
from mimic3models import common_utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
python -um mimic3models.length_of_stay.main --network mimic3models/pytorch_models/transformer.py --timestep 1.0 --mode train --batch_size 8 --partition custom --output_dir mimic3models/length_of_stay
"""

parser = argparse.ArgumentParser()
common_utils.add_common_arguments(parser)
parser.add_argument('--deep_supervision', dest='deep_supervision', action='store_true')
parser.set_defaults(deep_supervision=False)
parser.add_argument('--partition', type=str, default='custom',
                    help="log, custom, none")
parser.add_argument('--data', type=str, help='Path to the data of length-of-stay task',
                    default=os.path.join(os.path.dirname(__file__), '../../data/length-of-stay/'))
parser.add_argument('--output_dir', type=str, help='Directory relative which all output files are stored',
                    default='.')
parser.add_argument('--gpu_id', dest='gpu id', type=str, default='0')
args = parser.parse_args()
print(args)

device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')

if args.small_part:
    args.save_every = 2**30

# Build readers, discretizers, normalizers
if args.deep_supervision:
    train_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                               listfile=os.path.join(args.data, 'train_listfile.csv'),
                                                               small_part=args.small_part)
    val_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'train'),
                                                             listfile=os.path.join(args.data, 'val_listfile.csv'),
                                                             small_part=args.small_part)
else:
    train_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                      listfile=os.path.join(args.data, 'train_listfile.csv'))
    val_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'train'),
                                    listfile=os.path.join(args.data, 'val_listfile.csv'))

discretizer = Discretizer(timestep=args.timestep,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

if args.deep_supervision:
    discretizer_header = discretizer.transform(train_data_loader._data["X"][0])[1].split(',')
else:
    discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = args.normalizer_state
if normalizer_state is None:
    normalizer_state = 'los_ts{}.input_str:previous.start_time:zero.n5e4.normalizer'.format(args.timestep)
    normalizer_state = os.path.join(os.path.dirname(__file__), normalizer_state)
normalizer.load_params(normalizer_state)

args_dict = dict(args._get_kwargs())
args_dict['header'] = discretizer_header
args_dict['task'] = 'los'
args_dict['nhead'] = 32
args_dict['dim_feedforward'] = 128
args_dict['dropout'] = 0.5
args_dict['d_model'] = 2
args_dict['n_classes'] = (1 if args.partition == 'none' else 10)


# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
model.to(device)
suffix = "{}.bs{}{}{}.ts{}.partition={}".format("" if not args.deep_supervision else ".dsup",
                                                args.batch_size,
                                                ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                args.timestep,
                                                args.partition)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)

# Compile the model
print("==> compiling the model")
optimizer = optim.Adam(model.parameters(), lr=1e-3)
if args.partition == 'none':
    loss_function = nn.MSELoss() # 'mean_squared_error'
else:
    loss_function = nn.CrossEntropyLoss() # 'sparse_categorical_crossentropy'

# Load data and prepare generators
if args.deep_supervision:
    train_data_gen = utils.BatchGenDeepSupervision(train_data_loader, args.partition,
                                                   discretizer, normalizer, args.batch_size, shuffle=True)
    val_data_gen = utils.BatchGenDeepSupervision(val_data_loader, args.partition,
                                                 discretizer, normalizer, args.batch_size, shuffle=False)
else:
    # Set number of batches in one epoch
    train_nbatches = 2000
    val_nbatches = 1000
    if args.small_part:
        train_nbatches = 20
        val_nbatches = 20

    train_data_gen = utils.BatchGen(reader=train_reader,
                                    discretizer=discretizer,
                                    normalizer=normalizer,
                                    partition=args.partition,
                                    batch_size=args.batch_size,
                                    steps=train_nbatches,
                                    shuffle=True)
    val_data_gen = utils.BatchGen(reader=val_reader,
                                  discretizer=discretizer,
                                  normalizer=normalizer,
                                  partition=args.partition,
                                  batch_size=args.batch_size,
                                  steps=val_nbatches,
                                  shuffle=False)
if args.mode == 'train':
    # Prepare training
    path = os.path.join(args.output_dir, 'keras_states/' + model.final_name + '.chunk{epoch}.test{val_loss}.state')
    for _ in tqdm(range(args.epochs), desc='Training epoch: '):

        for batch in tqdm(train_data_gen, desc='Step: '):

            input_x, input_y, _ = tuple(t.to(device) for t in batch)
            pred = model(input_x)
            loss = loss_function(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

elif args.mode == 'test':
    # ensure that the code uses test_reader
    del train_data_gen
    del val_data_gen

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        del train_data_loader
        del val_data_loader
        test_data_loader = common_utils.DeepSupervisionDataLoader(dataset_dir=os.path.join(args.data, 'test'),
                                                                  listfile=os.path.join(args.data, 'test_listfile.csv'),
                                                                  small_part=args.small_part)
        test_data_gen = utils.BatchGenDeepSupervision(test_data_loader, args.partition,
                                                      discretizer, normalizer, args.batch_size,
                                                      shuffle=False, return_names=True)
        for i in range(test_data_gen.steps):
            print("\tdone {}/{}".format(i, test_data_gen.steps), end='\r')

            ret = test_data_gen.next(return_y_true=True)
            (x, y_processed, y) = ret["data"]
            cur_names = np.array(ret["names"]).repeat(x[0].shape[1], axis=-1)
            cur_ts = ret["ts"]
            for single_ts in cur_ts:
                ts += single_ts

            pred = model.predict(x, batch_size=args.batch_size)
            if pred.shape[-1] == 1:  # regression
                pred_flatten = pred.flatten()
            else:  # classification
                pred_flatten = pred.reshape((-1, 10))
            for m, t, p, name in zip(x[1].flatten(), y.flatten(), pred_flatten, cur_names.flatten()):
                if np.equal(m, 1):
                    labels.append(t)
                    predictions.append(p)
                    names.append(name)
    else:
        del train_reader
        del val_reader
        test_reader = LengthOfStayReader(dataset_dir=os.path.join(args.data, 'test'),
                                         listfile=os.path.join(args.data, 'test_listfile.csv'))
        test_data_gen = utils.BatchGen(reader=test_reader,
                                       discretizer=discretizer,
                                       normalizer=normalizer,
                                       partition=args.partition,
                                       batch_size=args.batch_size,
                                       steps=None,  # put steps = None for a full test
                                       shuffle=False,
                                       return_names=True)

        for i in range(test_data_gen.steps):
            print("predicting {} / {}".format(i, test_data_gen.steps), end='\r')

            ret = test_data_gen.next(return_y_true=True)
            (x, y_processed, y) = ret["data"]
            cur_names = ret["names"]
            cur_ts = ret["ts"]

            x = np.array(x)
            pred = model.predict_on_batch(x)
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)

    if args.partition == 'log':
        predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
        metrics.print_metrics_log_bins(labels, predictions)
    if args.partition == 'custom':
        predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
        metrics.print_metrics_custom_bins(labels, predictions)
    if args.partition == 'none':
        metrics.print_metrics_regression(labels, predictions)
        predictions = [x[0] for x in predictions]

    path = os.path.join(os.path.join(args.output_dir, "test_predictions", os.path.basename(args.load_state)) + ".csv")
    utils.save_results(names, ts, predictions, labels, path)

else:
    raise ValueError("Wrong value for args.mode")
