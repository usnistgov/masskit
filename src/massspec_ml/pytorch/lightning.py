import logging
import os
import random
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import open_dict
import resource
from massspec.utils.general import class_for_name


def get_pytorch_ranks():
    """
    get ranks for this process when training in parallel

    :return: is this training parallel?, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id
    """
    env_copy = os.environ.copy()
    if torch.utils.data.get_worker_info() is not None:
        worker_id = torch.utils.data.get_worker_info().id
        num_workers = torch.utils.data.get_worker_info().num_workers
    else:
        worker_id = 0
        num_workers = 1
    if env_copy.get("PL_IN_DDP_SUBPROCESS", "0") == "1":
        world_size = int(env_copy["WORLD_SIZE"])  # num_gpus * num_nodes
        num_gpus = len(env_copy["PL_TRAINER_GPUS"].split(","))
        num_nodes = world_size / num_gpus
        local_rank = int(env_copy["LOCAL_RANK"])  # rank on node
        node_rank = int(env_copy["NODE_RANK"])
        # subset the data into world_size pieces
        # take the node_rank*num_gpus + local_rank slice
        world_rank = node_rank * num_gpus + local_rank
        return True, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id
    else:
        return False, worker_id, num_workers, 1, 1, 0, worker_id, worker_id


def seed_worker():
    """
    set the random seed for this worker.  In future versions of pytorch lightning, can be replaced with
    pl_worker_init_function

    """
    is_parallel, world_rank, world_size, num_gpus, num_nodes, node_rank, local_rank, worker_id = get_pytorch_ranks()
    process_seed = torch.initial_seed()
    # back out the base seed so we can use all the bits
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([base_seed, worker_id, world_rank])
    # use 128 bits (4 x 32-bit words)
    np.random.seed(ss.generate_state(4))
    # Spawn distinct SeedSequences for the PyTorch PRNG and the stdlib random module
    torch_ss, stdlib_ss = ss.spawn(2)
    # PyTorch 1.7 and above takes a 64-bit seed
    dtype = np.uint64  # if _TORCH_GREATER_EQUAL_1_7 else np.uint32
    torch.manual_seed(torch_ss.generate_state(1, dtype=dtype)[0])
    # use 128 bits expressed as an integer
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)


def log_worker_start(worker_id):
    """
    function for initializing the Dataset

    :param worker_id: worker rank
    Notes:
      - since we are handling the sharding ourselves, it's necessary to disable adding of Distributed Sampler
       in Trainer by using replace_sampler_ddp=False
    """
    seed_worker()

    num_workers = torch.utils.data.get_worker_info().num_workers
    dataset = torch.utils.data.get_worker_info().dataset

    # now subset the data and slice it up for each worker
    # the assumption is that each worker receives batches of ids in the order of worker_id
    # note that we do not change dataset.length as pytorch wants this to be the number of rows of all data for a gpu
    batch_size = dataset.config.ml.batch_size
    dataset.worker_id = worker_id
    dataset.num_workers = num_workers
    """
    # subsetting the data by thread.  this was superceded by using pytorch samplers in the code below
    mask = np.array([False] * len(dataset.data.index))
    for i in range(len(mask)):
        if int(i / batch_size) % num_workers == worker_id:
            mask[i] = True

    dataset.data = dataset.data[mask].copy()

    if dataset.set_to_load == "train":
        # if requested in config, use a sampler to subset the data
        if "sampler_type" in dataset.config.ml.sampler and dataset.config.ml.sampler.sampler_type is not None:
            sampler = class_for_name(dataset.config.paths.modules.samplers,
                                     dataset.config.ml.sampler.sampler_type)(dataset)
            if sampler is not None:
                sampler.sample()

        # shuffle the training data if requested
        if dataset.config.ml.shuffle:
            dataset.data = dataset.data.sample(frac=1)
    """
    logging.debug(f'{len(dataset.data)} records in worker {worker_id}')
    logging.debug(
        f"started worker with info {torch.utils.data.get_worker_info()} and dataset {dataset.set_to_load}"
    )
    if resource:
        logging.debug(
            f"memory usage for worker {worker_id} is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB"
        )


class BaseDataModule(pl.LightningDataModule):
    """
    base class for data loading
    """

    def __init__(self, config, worker_init_fn=log_worker_start, *args, **kwargs):
        """
        :param config: config object
        :param worker_init_fn: function called to initialize each worker thread
        Notes:
            - for each worker, a duplicate Dataset is created via forking.  Then worker_init_fn is called and in
              the global torch.utils.data.get_worker_info(), dataset points to the copied Dataset
        """
        super().__init__(*args, **kwargs)
        self.config = config
        self.worker_init_fn = worker_init_fn

    def setup(self, stage=None):
        """
        called on every GPU

        :param stage: is set to "fit" or "test"
        :return: self
        """
        return self

    def create_loader(self, set_to_load=None):
        pass

    def train_dataloader(self):
        return self.create_loader("train")

    def val_dataloader(self):
        return self.create_loader("valid")

    def test_dataloader(self):
        return self.create_loader("test")


class XORDataModule(BaseDataModule):
    """
    data loader for XOR toy network
    """

    def __init__(self, config, worker_init_fn=log_worker_start, *args, **kwargs):
        """
        :param config: config object
        :param worker_init_fn: function called to initialize each worker thread
        Notes:
            - for each worker, a duplicate Dataset is created via forking.  Then worker_init_fn is called and in
              the global torch.utils.data.get_worker_info(), dataset points to the copied Dataset
        """
        super().__init__(config, worker_init_fn=log_worker_start, *args, **kwargs)

    def create_loader(self, set_to_load=None):
        # subset will be dataset for xor
        df = pd.DataFrame({'input': [np.array([0.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32),
                                     np.array([1.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)]
                                    * 100,
                           'output': [np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32),
                                      np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32)] * 100})
        subset = class_for_name(self.config.paths.modules.dataloaders,
                                self.config.ms.dataloader)(df, self.config)
        # set the length of the dataset.  Used in bayes computation to normalize KL divergence
        if 'num' in self.config.input[set_to_load]:
            self.config.input[set_to_load].num = len(subset.data)
        else:
            with open_dict(self.config):
                self.config.input[set_to_load].num = len(subset.data)
        return torch.utils.data.DataLoader(
            dataset=subset,
            num_workers=self.config.setup.num_workers,
            batch_size=self.config.ml.batch_size,
            worker_init_fn=self.worker_init_fn,
            pin_memory=True
        )