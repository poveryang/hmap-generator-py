import os
from argparse import Namespace

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from utils import next_version

# ---------- Configs ----------
home_path = os.path.expanduser('~')
log_dir = os.path.join(home_path, 'ExpLogs')
exp_name = 'UNet'
version = next_version(log_dir, exp_name)

conf = Namespace(data=Namespace(), model=Namespace(), train=Namespace())

# data conf
conf.data.batch_size = 16
conf.data.num_workers = 8
conf.data.root_dir = f'{home_path}/Datasets/barcode'
conf.data.input_size = (400, 640)

# model conf
conf.model.in_channels = 1
conf.model.n_classes = 3
conf.model.inc_channels = 16
conf.model.init_lr = 1e-3

# train conf
conf.train.accelerator = 'gpu'
conf.train.devices = [0, 1, 2, 3, 4, 5, 6, 7]
conf.train.sync_batchnorm = True
conf.train.max_epochs = 100
conf.train.profiler = 'simple'
conf.train.default_root_dir = log_dir
conf.train.limit_train_batches = 1.
conf.train.limit_val_batches = 1.
conf.train.strategy = DDPStrategy(find_unused_parameters=False)
conf.train.logger = TensorBoardLogger(log_dir, name=exp_name, version=f'v{version}')
# ---------- Configs ----------
