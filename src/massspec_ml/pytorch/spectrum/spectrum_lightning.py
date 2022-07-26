from abc import ABC
import importlib
from omegaconf import ListConfig
import pytorch_lightning as pl
from massspec_ml.pytorch.lightning import log_worker_start, BaseDataModule
from massspec_ml.pytorch.samplers import DistributedSamplerWrapper
from torch.utils.data import WeightedRandomSampler
from urllib.parse import urlparse
from urllib import request
from pathlib import Path
from massspec_ml.pytorch.spectrum.peptide.peptide_embed import *
from massspec.utils.general import class_for_name, search_for_file
from massspec_ml.pytorch.lightning import seed_worker
from massspec_ml.pytorch.spectrum.spectrum_datasets import TandemArrowDataset
import pyarrow as pa
from pyarrow import plasma
from pyarrow.plasma import ObjectID
from massspec.utils.index import ArrowLibraryMap
import logging
import hashlib
import builtins

try:
    import boto3
except ImportError:
    logging.debug("Unable to import boto3")
    boto3 = None

try:
    import resource
except ImportError:
    logging.debug("Unable to import resource")
    resource = None


class BaseSpectrumLightningModule(pl.LightningModule, ABC):
    """
    base class for pytorch lightning module used to run the training
    """

    def __init__(
            self, config=None, *args, **kwargs
    ):
        """
        :param config: configuration dictionary
        """
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(config)
        self.config = config

        model_class = class_for_name(self.config.paths.modules.models, list(self.config.ml.model.keys())[0])
        self.model = model_class(config)

        self.loss_function = class_for_name(self.config.paths.modules.losses,
                                            self.config.ml.loss.loss_function
                                            )(config=self.config)

        # create a list of metrics
        # for training
        self.train_metrics = {}
        # for validation and test
        self.valid_metrics = {}
        if "metrics" in self.config.ml and self.config.ml.metrics is not None:
            for metric in self.config.ml.metrics:
                self.train_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
                self.valid_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
        if "valid_metrics" in self.config.ml and self.config.ml.valid_metrics is not None:
            for metric in self.config.ml.valid_metrics:
                self.valid_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)
        if "train_metrics" in self.config.ml and self.config.ml.train_metrics is not None:
            for metric in self.config.ml.train_metrics:
                self.train_metrics[metric] = class_for_name(self.config.paths.modules.metrics,
                                                            metric)(config=self.config)

    def forward(self, x):
        return self.model(x)

    def calc_loss(self, output, batch, params=None):
        """
        overrideable loss function

        :param output: output from the model
        :param batch: batch data, including input and true spectra
        :param params: optional dictionary of parameters, such as epoch type
        :return: loss
        """
        return self.loss_function(output, batch, params=params)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        """
        validation step

        :param batch: batch data tensor
        :param batch_idx: the index of the batch
        :param dataloader_idx: which dataloader is being used (None if just one)
        :return: loss
        """
        return self.validation_test_step(batch, batch_idx, 'valid')

    def test_step(self, batch, batch_idx):
        return self.validation_test_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        return getattr(importlib.import_module('torch.optim'),
                       self.config.ml.optimizer.optimizer_function)(self.model.parameters(),
                                                                    lr=self.config.ml.optimizer.lr)

    # lightning appears to be in the process of redoing logging, and this function will
    # throw an exceptions with some versions of lightning
    # def custom_histogram_adder(self):
    #     # iterating through all parameters
    #     # lightning sometimes add the tensorboard logger itself, so handle this
    #     try:
    #         loggers = iter(self.logger)
    #     except TypeError:
    #         loggers = iter([self.logger])

    #     for logger in loggers:
    #         if isinstance(logger, pl.loggers.tensorboard.TensorBoardLogger):
    #             for name, params in self.model.named_parameters():
    #                 logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_step_end(self, outputs):
        if isinstance(outputs, torch.Tensor):
            loss = outputs
        else:
            loss = torch.stack(outputs).mean()
        if self.global_step % 100 == 0:
            self.log("loss", loss, on_step=True)
        # self.logger.log_metrics({"loss": loss.item()})
        return loss

    def training_epoch_end(self, outputs):
        output = f"Epoch {self.trainer.current_epoch}: train loss={float(self.trainer.progress_bar_dict['loss']):g}"
        # logging histograms
        # self.custom_histogram_adder()
        self.trainer.progress_bar_callback.main_progress_bar.write(output)
        for metric, metric_function in self.train_metrics.items():
            metric_function.reset()

    def validation_test_epoch_end(self, outputs, loop):
        """
        shared code between validation and test epoch ends.  logs and prints losses, resets metrics

        :param outputs: output list from model
        :param loop: 'val' or 'test'
        """
        # single validation set returns a list of tensors, multi validation returns a list of list of tensors
        # if single validation, insert it into a list
        if not isinstance(outputs[0], list):
            outputs = [outputs]
        losses = []
        for i, output in enumerate(outputs):
            losses.append(torch.stack(output).mean().item())
        self.log(f'{loop}_loss', losses[0])
        for i in range(1, len(losses)):
            self.log(f'{loop}_loss_{i}', losses[i])
        output = f"Epoch {self.trainer.current_epoch}: {loop} loss={','.join([f'{x:g}' for x in losses])}"
        self.trainer.progress_bar_callback.main_progress_bar.write(output)
        for metric, metric_function in self.valid_metrics.items():
            metric_function.reset()

    def validation_epoch_end(self, outputs):
        self.validation_test_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self.validation_test_epoch_end(outputs, 'test')


class SpectrumLightningModule(BaseSpectrumLightningModule):
    """
    pytorch lightning module used to run the training
    """

    def __init__( self, config=None, *args, **kwargs):
        """
        :param config: configuration dictionary
        It's important that the config parameter is explictly defined so lightning serialization works
        """
        super().__init__(config=config, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.config.ml.bayesian_network.bayes:
            loss = 0.0
            for step in range(self.config.ml.bayesian_network.sample_nbr):
                output = self.model(batch)
                loss += self.calc_loss(output, batch, params={'loop': 'train'})
            loss /= self.config.ml.bayesian_network.sample_nbr
        else:
            output = self.model(batch)
            loss = self.calc_loss(output, batch, params={'loop': 'train'})

        for metric, metric_function in self.train_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(output, batch)
            self.log(f'training_' + metric, metric_value, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_test_step(self, batch, batch_idx, loop):
        """
        step shared with test and validation loops

        :param batch: batch
        :param batch_idx: index into data for batch
        :param loop: the name of the loop
        :return: loss
        """
        if self.config.ml.bayesian_network.bayes:
            loss = 0.0
            for step in range(self.config.ml.bayesian_network.sample_nbr):
                output = self.model(batch)
                loss += self.calc_loss(output, batch, params={'loop': loop})
            loss /= self.config.ml.bayesian_network.sample_nbr
        else:
            output = self.model(batch)
            loss = self.calc_loss(output, batch, params={'loop': loop})

        for metric, metric_function in self.valid_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(output, batch)
            self.log(f'{loop}_' + metric, metric_value, prog_bar=True, on_step=False, on_epoch=True)
        return loss


def log_worker_start_spectrum(worker_id):
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

    dataset.init_copy(worker_id=worker_id, num_workers=num_workers)

    logging.debug(
        f"started worker with info {torch.utils.data.get_worker_info()} and dataset {dataset.set_to_load}"
    )
    if resource:
        logging.debug(
            f"memory usage for worker {worker_id} is {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss} kB"
        )

"""
base class for spectra data modules
"""


class SpectrumDataModule(BaseDataModule):
    """
    data loader for tandem spectra
    """

    def __init__(self, config, worker_init_fn=log_worker_start_spectrum, *args, **kwargs):
        """
        :param config: config object
        :param worker_init_fn: function called to initialize each worker thread
        Notes:
            - for each worker, a duplicate Dataset is created via forking.  Then worker_init_fn is called and in
              the global torch.utils.data.get_worker_info(), dataset points to the copied Dataset
        """
        super().__init__(config, worker_init_fn=worker_init_fn, *args, **kwargs)

    def create_loader(self, set_to_load=None):
        """
        helper function to load data

        :param set_to_load: name of the set to load
        :return: loader or list of loaders
        """

        subsets = self.get_subsets(set_to_load)

        # set the length of the dataset.  Used in bayes computation to normalize KL divergence
        self.config.input[set_to_load].num = sum([len(subset) for subset in subsets])

        if set_to_load == 'train':
            # ConcatDataset(subsets) currently not implemented as ConcatDataset is not derived from BaseDataset
            subset = subsets[0]
            if self.config.ml.get("sampler", None) is not None and self.config.ml.sampler.get("sampler_type", None) is not None:
                sampler = class_for_name(self.config.paths.modules.samplers,
                                         self.config.ml.sampler.sampler_type)(subset)

                probabilities = sampler.probability()
                pytorch_sampler = WeightedRandomSampler(probabilities, len(probabilities), replacement=True)

            elif self.config.ml.shuffle:
                pytorch_sampler = WeightedRandomSampler([1.0] * len(subset), len(subset), replacement=False)
            else:
                pytorch_sampler = None

            # if using ddp, use the distributed sampler wrapper
            # this needs to be tested carefully.  In particular, it's not clear how DistributedSamplerWrapper
            # knows how to cut up the data into multiple workers as it is initialized before the workers
            # most likely it is called before the workers are init'ed and the sampling list is then
            # cut up for the workers
            # note that DistributedSamplerWrapper requires that ddp be initialized via
            # torch.distributed.init_process_group, which is presumably called by pytorch lightning.
            if self.config.setup.accelerator is not None:
                pytorch_sampler = DistributedSamplerWrapper(pytorch_sampler)

            return torch.utils.data.DataLoader(
                dataset=subset,
                num_workers=self.config.setup.num_workers,
                batch_size=self.config.ml.batch_size,
                worker_init_fn=self.worker_init_fn,
                pin_memory=True,
                sampler=pytorch_sampler
            )
        else:
            # for validation and test sets, return a list of Dataloaders as each is evaluated independently
            return [torch.utils.data.DataLoader(
                dataset=subset,
                num_workers=self.config.setup.num_workers,
                batch_size=self.config.ml.batch_size,
                worker_init_fn=self.worker_init_fn,
                pin_memory=True
            ) for subset in subsets]
            
    def get_subsets(self, set_to_load):
        """
        create datasets

        :param set_to_load: train, valid or test dataset
        :return: a list of datasets
        """
        # check to see if there is a list of spectral libraries
        if isinstance(self.config.input[set_to_load].spectral_library, ListConfig):
            spectral_libraries = self.config.input[set_to_load].spectral_library
        else:
            spectral_libraries = [self.config.input[set_to_load].spectral_library]

        subsets = []
        for spectral_library in spectral_libraries:
            path = self.get_dataset_path(spectral_library)
            logging.debug(
                f"SpectrumDataModule create_loader for {set_to_load} called for db {path}"
            )
            subsets.append(class_for_name(self.config.paths.modules.dataloaders,
                                    self.config.ms.dataloader)(path, self.config, set_to_load))
        return subsets

    def get_dataset_path(self, spectral_library):
        """
        for a given spectral_library, return the path (downloading file if necessary)

        :param spectral_library: name of the spectral library
        :return: path
        """
        url = urlparse(spectral_library, allow_fragments=False)
        if url.scheme in ["s3", "http", "https"]:
            path = search_for_file(url.path.lstrip("/"), self.config.paths.search_path)
            # if the file doesn't exist in the cache, download it
            if path is None or not path.is_file():
                path = Path(self.config.paths.cache_directory, url.path.lstrip("/"))
                path.parent.mkdir(
                    parents=True, exist_ok=True
                )  # make the cache directory
                if url.scheme == "s3":
                    s3 = boto3.client("s3")
                    with open(path, "wb") as f:
                        s3.download_fileobj(url.netloc, url.path.lstrip("/"), f)
                else:
                    request.urlretrieve(spectral_library, path)
        else:
            path = Path(spectral_library)

        return path

