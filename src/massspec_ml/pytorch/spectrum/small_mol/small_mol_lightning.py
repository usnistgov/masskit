from general import class_for_name
from omegaconf import ListConfig
import torch
from massspec_ml.pytorch.base_objects import ModelInput, ModelOutput
from massspec_ml.pytorch.spectrum.spectrum_lightning import BaseSpectrumLightningModule, SpectrumDataModule, log_worker_start_spectrum
import logging


class SmallMolSearchDataModule(SpectrumDataModule):
    """
    data loader for tandem small molecule search
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
        
    def get_subsets(self, set_to_load):
        # check to see if there is a list of spectral libraries
        if isinstance(self.config.input[set_to_load].spectral_library, ListConfig):
            raise NotImplementedError('multiple datasets not supported for SmallMolSearchDataModule')

        subsets = []
        path = self.get_dataset_path(self.config.input[set_to_load].spectral_library)
        path_search = self.get_dataset_path(self.config.input[set_to_load].spectral_library_search)
        subsets.append(class_for_name(self.config.paths.modules.dataloaders,
                                self.config.ms.dataloader)(path, self.config, set_to_load, store_search=path_search))
        return subsets


class SearchLightningModule(BaseSpectrumLightningModule):
    """
    pytorch lightning module used to train on search results
    """

    def __init__(
            self, config=None, *args, **kwargs
    ):
        """
        :param config: configuration dictionary.  
        It's important that the config parameter is explictly defined so lightning serialization works
        """
        super().__init__(config=config, *args, **kwargs)

    def pairwise_search_model(self, batch):
        """
        feed batched hitlist data and corresponding spectra into the model.
        accomplishes this by reshaping data with form 
        (batch, hitlist, query/hit, spectrum) into (batch, spectrum) vectors
        with model output (batch, fingerprint) converted to (batch, hitlist, query/hit, fingerprint) 
        cosine score is then computed between query/hit pairs of fingerprints

        :param batch: input to the batch
        :param output: expected output
        :return: results from the model
        """
        x_shape = batch.x.shape
        # reshape so it acts like a batch of spectra
        xx = batch.x.view(x_shape[0] * x_shape[1] * x_shape[2], x_shape[3])
        # create a new input object
        new_input = ModelInput(x=xx, y=batch.y, index=batch.index)
        # (batch, hitlist, query/hit, spectrum) -> (batch*hitlist*2, spectrum)
        output = self.model(new_input)
        y_shape = output.y_prime.shape
        yy = output.y_prime.view(x_shape[0], x_shape[1], x_shape[2], y_shape[-1])
        dot_prod = torch.nn.functional.cosine_similarity(yy[:,:,0,:], yy[:,:,1,:], dim=-1)
        # dot_prod should be (batch, hitlist)
        new_output = ModelOutput(y_prime=dot_prod, score=output.score, var=output.var)
        return new_output

    def training_step(self, batch, batch_idx):
        # batch is, I think, (batch, ModelInput)?
        # batch should contain search results, so ModelInput.x has (hitlist, query/hit, spectrum)
        # iterate through hitlist, send query to model and hit to model
        # dot product each query and hit
        # send dot product to loss as (batch, hitlist)
        # loss is returned as (batch, hitlist)
        # loss should ignore values with tanimoto = 0.0
        new_output = self.pairwise_search_model(batch)
        loss = self.calc_loss(new_output, batch, params={'loop': 'train'})

        for metric, metric_function in self.train_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(new_output, batch)
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
        new_output = self.pairwise_search_model(batch)
        loss = self.calc_loss(new_output, batch, params={'loop': loop})

        for metric, metric_function in self.valid_metrics.items():
            # required to evaluate metric in each batch
            metric_value = metric_function(new_output, batch)
            self.log(f'{loop}_' + metric, metric_value, prog_bar=True, on_step=False, on_epoch=True)
        return loss

# batch.x.shape torch.Size([100, 10, 2, 20000])
# batch.y.shape torch.Size([100, 10])
# batch.index torch.Size([100])