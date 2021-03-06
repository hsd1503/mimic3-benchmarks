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
from mimic3models import common_utils

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

"""
python -um mimic3models.length_of_stay.main_pytorch --network mimic3models/pytorch_models/transformer.py --timestep 1.0 --mode train --batch_size 1 --partition custom --output_dir mimic3models/length_of_stay/exp --gpu_id 4 --run_id 0 --prefix new_


python -um mimic3models.length_of_stay.main_pytorch --network mimic3models/pytorch_models/transformer.py --timestep 1.0 --mode test --batch_size 8 --partition custom --output_dir mimic3models/length_of_stay/exp --gpu_id 4 --run_id 0
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
args_dict['nhead'] = 8
args_dict['dim_feedforward'] = 128
args_dict['dropout'] = 0.5
args_dict['d_model'] = 76
args_dict['d_emb'] = 128
args_dict['verbose'] = False
args_dict['n_classes'] = (1 if args.partition == 'none' else 10)


# Build the model
print("==> using model {}".format(args.network))
model_module = imp.load_source(os.path.basename(args.network), args.network)
model = model_module.Network(**args_dict)
model.to(device)
suffix = "{}.bs{}{}{}.ts{}.partition_{}.run{}".format("" if not args.deep_supervision else ".dsup",
                                                args.batch_size,
                                                ".L1{}".format(args.l1) if args.l1 > 0 else "",
                                                ".L2{}".format(args.l2) if args.l2 > 0 else "",
                                                args.timestep,
                                                args.partition,
                                                args.run_id)
model.final_name = args.prefix + model.say_name() + suffix
print("==> model.final_name:", model.final_name)
save_path = os.path.join(args.output_dir, model.final_name)
os.makedirs(save_path, exist_ok=True)

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
    train_nbatches = 500
    val_nbatches = 500
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
    step = 0
    all_loss = []
    best_val = 1e9
    
    print('start training, will save to {}'.format(save_path))
    with open(os.path.join(save_path, 'log.txt'), 'w') as fout:
        print(common_utils.get_time_str(), file=fout)    
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10)
    for i_epoch in tqdm(range(args.epochs), desc='Training epoch: '):

        for i_step in tqdm(range(train_data_gen.steps)):

            ### training
            model.train()
            input_x, input_y, _ = train_data_gen.next(return_y_true=True)
            input_x = torch.tensor(input_x, dtype=torch.float).to(device)
            input_y = torch.tensor(input_y, dtype=torch.long).to(device)

            pred = model(input_x)
            loss = loss_function(pred, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            all_loss.append(loss.item())
            if i_step % train_nbatches == 0:
                print(np.mean(all_loss[:-train_nbatches:]))

            ### validation
            if step % val_nbatches == 0:
                model.eval()

                y_true = []
                predictions = []
                for _ in tqdm(range(val_data_gen.steps)):
                    x, _, y = val_data_gen.next(return_y_true=True)
                    x = torch.tensor(x, dtype=torch.float).to(device)
                    pred = model(x)
                    pred = pred.cpu().data.numpy()
                    if isinstance(x, list) and len(x) == 2:  # deep supervision
                        pass
                    else:
                        if pred.shape[-1] == 1:
                            y_true += list(y.flatten())
                            predictions += list(pred.flatten())
                        else:
                            y_true += list(y)
                            predictions += list(pred)
                print('\n')
                if args.partition == 'log':
                    predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
                    ret = metrics.print_metrics_log_bins(y_true, predictions)
                if args.partition == 'custom':
                    predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
                    ret = metrics.print_metrics_custom_bins(y_true, predictions)
                if args.partition == 'none':
                    ret = metrics.print_metrics_regression(y_true, predictions)
                cur_val = ret['mse']
                
                scheduler.step(cur_val)
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < 1e-5:
                    with open(os.path.join(save_path, 'log.txt'), 'a') as fout:
                        print('Early stop at step {}'.format(step), file=fout)
                    exit()
                
                with open(os.path.join(save_path, 'log.txt'), 'a') as fout:
                    print(ret, file=fout)
                is_best = cur_val < best_val
                if is_best:
                    best_val = cur_val
                    ### save model
                    common_utils.save_checkpoint({
                        'epoch': i_epoch,
                        'step': step,
                        'state_dict': model.state_dict(),
                        'best_val': best_val,
                        'res': ret,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, path=save_path)
                
                if args.small_part:
                    exit()

elif args.mode == 'test':
    # ensure that the code uses test_reader
    del train_data_gen
    del val_data_gen
    
    ### load model
    checkpoint = torch.load(os.path.join(save_path, 'best_checkpoint.pth'))
    model.load_state_dict(checkpoint['state_dict'])
    print('start testing, load best_checkpoint.pth from {}'.format(save_path))

    names = []
    ts = []
    labels = []
    predictions = []

    if args.deep_supervision:
        ### not implement yet
        pass
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
            (x, _, y) = ret["data"]
            cur_names = ret["names"]
            cur_ts = ret["ts"]

            x = torch.tensor(x, dtype=torch.float).to(device)
            pred = model(x)
            pred = pred.cpu().data.numpy()
            predictions += list(pred)
            labels += list(y)
            names += list(cur_names)
            ts += list(cur_ts)
            
            if args.small_part:
                break

    if args.partition == 'log':
        predictions = [metrics.get_estimate_log(x, 10) for x in predictions]
        metrics.print_metrics_log_bins(labels, predictions)
    if args.partition == 'custom':
        predictions = [metrics.get_estimate_custom(x, 10) for x in predictions]
        metrics.print_metrics_custom_bins(labels, predictions)
    if args.partition == 'none':
        metrics.print_metrics_regression(labels, predictions)
        predictions = [x[0] for x in predictions]

    utils.save_results(names, ts, predictions, labels, os.path.join(save_path, 'test_predictions.csv'))

else:
    raise ValueError("Wrong value for args.mode")
