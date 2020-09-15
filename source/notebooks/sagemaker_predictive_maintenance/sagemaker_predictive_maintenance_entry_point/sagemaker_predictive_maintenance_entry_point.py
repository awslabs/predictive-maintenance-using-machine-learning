import os
import json
import time
import argparse
import logging
import random

import mxnet as mx
from mxnet import gluon, autograd, nd
import gluonnlp
from gluonnlp.data.batchify import Pad, Stack, Tuple
import pandas as pd
import numpy as np


def get_logger(name):
    logger = logging.getLogger(name)
    log_format = '%(asctime)s %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    return logger


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-datasets', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--log-interval', type=int, default=1000)
    parser.add_argument('--is-many-to-one', type=bool, default=False)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--num-units', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', type=float, default=0.00001)
    parser.add_argument('--clip-gradient', type=float, default=20)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    return parser.parse_args()


def read_data(training_dir, num_datasets):
    train_df = [pd.read_csv(os.path.join(training_dir, 'train-{}.csv'.format(i))) for i in range(num_datasets)]
    return train_df


class CombinedDataset(gluon.data.Dataset):
    """
    A dataset that accepts several dataset and serves
    them as one
    """

    def __init__(self, datasets):

        self.datasets = datasets

        self.lengths = []
        start = 0
        for d in datasets:
            end = start + len(d)
            self.lengths.append((start, end))
            start = end

        self.length = sum([len(d) for d in datasets])

    def __getitem__(self, idx):
        current_running = 0
        for i, (start, end) in enumerate(self.lengths):
            print(start, end, idx)
            if idx >= end:
                current_running += end
            else:
                return self.datasets[i][idx - current_running]

    def __len__(self):
        return self.length


class PredictiveMaintenanceDataset(gluon.data.Dataset):

    def __init__(self, dataframe, is_train=True, is_many_to_one=True):
        self.dataframe = dataframe
        self.is_train = is_train
        self.is_many_to_one = is_many_to_one

    def __getitem__(self, idx):
        data = np.array(self.dataframe[self.dataframe['id'] == idx + 1][self.dataframe.columns[2:-1]])
        label = np.array(self.dataframe[self.dataframe['id'] == idx + 1]['RUL'])
        if self.is_train:
            start = random.randint(0, data.shape[0] // 2 - 2)
            end = random.randint(data.shape[0] // 2 + 2, data.shape[0] - 1)
        else:
            start = 0
            end = data.shape[0] - 1

        data = data[start:end, :].astype('float32')
        label = label[start].astype('float32') if self.is_many_to_one else label[start:end].astype('float32')

        # Max RUL
        max_rul = 130.0
        if not self.is_many_to_one:
            label[label > max_rul] = max_rul
        else:
            label = min(label, max_rul)
            
        label = label.astype('float32')

        # Duplicate first element to avoid cliff-edge
        # data = np.concatenate((np.expand_dims(data[0], axis=0), data), axis=0)
        # label = np.concatenate((np.expand_dims(label[0], axis=0), label), axis=0)
        return data, label

    def __len__(self):
        return len(self.dataframe['id'].unique())


def RMSE_many_to_one(predictions, labels, data_lengths):
    # predictions = predictions.sum(axis=1).squeeze() / data_lengths.astype('float32')
    loss = (predictions - labels).square()
    return loss


def RMSE_many_to_many(predictions, labels, data_lengths):
    loss = (labels.expand_dims(axis=2) - predictions).square()
    loss = nd.SequenceMask(loss, data_lengths, use_sequence_length=True, axis=1)
    weight = 1 / (labels + 1)
    loss_no_weight = loss.sum(axis=1).squeeze() / data_lengths.astype('float32')
    loss_weighted = ((loss.squeeze() * weight).sum(axis=1).squeeze() / data_lengths.astype('float32'))
    return loss_weighted, loss_no_weight


class TimeSeriesNet(gluon.nn.HybridBlock):

    def __init__(self, num_layers, num_units, dropout):
        super(TimeSeriesNet, self).__init__()
        
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout = dropout

        with self.name_scope():
            self.net = gluon.nn.HybridSequential(prefix='predictive_maintenance_')
            with self.net.name_scope():
                self.net.add(
                    gluon.nn.HybridLambda(lambda F, x: x.transpose((0, 2, 1))),
                    gluon.nn.Conv1D(channels=32, kernel_size=3, padding=1),
                    gluon.nn.Conv1D(channels=32, kernel_size=3, padding=1),
                    gluon.nn.HybridLambda(lambda F, x: x.transpose((0, 2, 1))),
                    gluon.rnn.LSTM(num_units, num_layers=num_layers, bidirectional=True, layout='NTC', dropout=dropout,
                                   state_clip_min=-10, state_clip_max=10, state_clip_nan=True),
                    gluon.nn.Activation('softrelu'),
                )
            self.proj = gluon.nn.Dense(1, flatten=False)

    def hybrid_forward(self, F, x):
        return self.proj(self.net(x))


def train(net, train_dataloader, epochs, batch_size, is_many_to_one, model_dir):
    loss_fn = RMSE_many_to_one if is_many_to_one else RMSE_many_to_many
    INPUT_SCALER = 300

    for e in range(epochs):
        loss_avg = 0
        for i, ((data, data_lengths), (label)) in enumerate(train_data):
            data = data.as_in_context(ctx).astype('float32')
            label = label.as_in_context(ctx).astype('float32')
            data_lengths = data_lengths.as_in_context(ctx).astype('float32')
            with autograd.record():
                pred = net(data)
                loss, loss_no_weight = loss_fn(pred, label / INPUT_SCALER, data_lengths)
                loss = loss.mean()
            loss.backward()
            trainer.step(data.shape[0])
            loss_avg += loss_no_weight.mean().sqrt().asnumpy()
        logging.info("Epoch {}: Average RMSE {}".format(e, INPUT_SCALER * loss_avg / (i + 1)))
    
    save_model(net, model_dir)
    logging.info("Saved model params")
    logging.info("End of training")

def save_model(net, model_dir):
    net.save_parameters(os.path.join(model_dir, "net.params"))
    f = open(os.path.join(model_dir, "model.params"), 'w')
    json.dump({'num_layers': net.num_layers,
               'num_units': net.num_units,
               'dropout': net.dropout},
              f)
    f.close()

if __name__ == '__main__':
    logging = get_logger(__name__)
    logging.info('numpy version:{} MXNet version::{}'.format(np.__version__, mx.__version__))
    options = parse_args()

    ctx = mx.gpu() if options.num_gpus > 0 else mx.cpu()

    train_df = read_data(options.training_dir, options.num_datasets)

    train_datasets = [PredictiveMaintenanceDataset(df, is_train=True, is_many_to_one=options.is_many_to_one) for df in
                      train_df]

    batchify = Tuple(Pad(ret_length=True), Stack() if options.is_many_to_one else Pad())

    dataset_index = 0
    train_data = gluon.data.DataLoader(train_datasets[dataset_index], shuffle=True, batch_size=options.batch_size,
                                       num_workers=8,
                                       batchify_fn=batchify)

    logging.info("We have {} training timeseries".format(len(train_datasets[dataset_index])))

    net = TimeSeriesNet(options.num_layers, options.num_units, options.dropout)
    net.hybridize(static_alloc=True)
    net.initialize(mx.init.Normal(), ctx=ctx)
    logging.info('Model created and initialized')

    optimizer_params = {'learning_rate': options.learning_rate, 'wd': options.wd,
                        'clip_gradient': options.clip_gradient}

    if options.optimizer == 'sgd':
        optimizer_params['momentum'] = options.momentum

    trainer = gluon.Trainer(net.collect_params(), options.optimizer, optimizer_params)

    train(net, train_data, options.epochs, options.batch_size, options.is_many_to_one, options.model_dir)
    

class TimeSeriesNetInfer(gluon.nn.HybridBlock):

    def __init__(self, num_layers, num_units, dropout):
        super(TimeSeriesNetInfer, self).__init__()
        
        self.num_layers = num_layers
        self.num_units = num_units
        self.dropout = dropout

        with self.name_scope():
            self.net = gluon.nn.HybridSequential(prefix='predictive_maintenance_')
            with self.net.name_scope():
                self.net.add(
                    gluon.nn.HybridLambda(lambda F, x: x.transpose((0, 2, 1))),
                    gluon.nn.Conv1D(channels=32, kernel_size=3, padding=1),
                    gluon.nn.Conv1D(channels=32, kernel_size=3, padding=1),
                    gluon.nn.HybridLambda(lambda F, x: x.transpose((0, 2, 1))),
                    gluon.rnn.LSTM(num_units, num_layers=num_layers, bidirectional=True, layout='NTC', dropout=dropout),
                    gluon.nn.Activation('softrelu'),
                )
            self.proj = gluon.nn.Dense(1, flatten=False)

    def hybrid_forward(self, F, x):
        return self.proj(self.net(x))
    
def model_fn(model_dir):
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    with open(os.path.join(model_dir, "model.params"), 'r') as f:
        net_params = json.load(f)
     
    net = TimeSeriesNetInfer(net_params['num_layers'], net_params['num_units'], net_params['dropout'])
    net.load_parameters(os.path.join(model_dir, "net.params"), ctx)
    return net
    
def transform_fn(net, data, input_content_type, output_content_type):
    data_dict = json.loads(data.decode())
    input_data = nd.array(data_dict['input'])
    
    ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    input_data = input_data.as_in_context(ctx)
    pred = net(input_data)
    
    response = json.dumps(pred.asnumpy().tolist())
    return response, output_content_type
