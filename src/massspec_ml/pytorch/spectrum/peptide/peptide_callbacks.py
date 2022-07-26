import datetime
import os
import tempfile

import numpy as np
import pytorch_lightning as pl
import torch

from massspec.spectrum.plotting import AnimateSpectrumPlot, multiple_spectrum_plot


class PeptideCB(pl.callbacks.base.Callback):
    """
    callbacks for peptide spectra training
    """

    def __init__(self, config, loggers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loggers = loggers
        self.config = config
        if self.config.logging.images.animate:
            self.animator = AnimateSpectrumPlot()
        else:
            self.animator = None

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx == 0 and (dataloader_idx == 0 or dataloader_idx is None) and self.loggers and self.config.logging.image_epoch_interval:

            # pytorch lightning renamed running_sanity_check to sanity_checking in 1.5
            sanity_checking = False
            if hasattr(trainer, 'running_sanity_check') and trainer.running_sanity_check:
                sanity_checking = True
            if hasattr(trainer, 'sanity_checking') and trainer.sanity_checking:
                sanity_checking = True

            if sanity_checking or trainer.current_epoch % self.config.logging.image_epoch_interval:
                return
            with torch.no_grad():
                # predict the spectra using the model
                pl_module.eval()
                # send the input tensors to the gpu
                for key in batch._fields:
                    value = getattr(batch, key)
                    if isinstance(value, torch.Tensor):
                        batch = batch._replace(**{key: value.to(pl_module.device)})
                predicted_spectra = pl_module(batch)
                pl_module.train()
                predicted_spectra = predicted_spectra.y_prime[
                    0 : self.config.logging.images.num_images, 0, :
                ]

                spectra = torch.squeeze(batch.y)[0 : self.config.logging.images.num_images, :]
                # generate mz values
                mz = np.linspace(0, self.config.ms.max_mz, spectra.shape[-1], endpoint=False)

                # create a mirror plot and log
                title = (
                    f"Validation epoch: {trainer.current_epoch}, "
                    f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    f"\nExperiment: {self.config.experiment_name}"
                )
                if "MSMLFlowLogger" in self.loggers:
                    title += f"\nRun: {self.loggers['MSMLFlowLogger'].run_id}"

                subtitles = []
                for index in batch.index:
                    data = trainer.val_dataloaders[0].dataset.get_data_row(index)
                    if len(data['peptide']) > 15:
                        name = data['peptide'][0: 15 - 1] + ">"
                    else:
                        name = data['peptide']
                    subtitles.append(f"{name} +{data['charge']} {int(data['ev'])}eV")

                fig = multiple_spectrum_plot(
                    spectra.cpu().numpy(),
                    mirror_intensities=predicted_spectra.cpu().numpy(),
                    dpi=self.config.logging.images.dpi,
                    mz=mz,
                    min_mz=0,
                    max_mz=self.config.ms.max_mz,
                    title=title,
                    subtitles=subtitles,
                    normalize=self.config.logging.images.intensity_norm
                )

                # create a tag for the logging system
                filename = f"val_epoch_{trainer.current_epoch:03d}.gif"
                # log the figure
                if self.animator:
                    self.animator.add_figure(fig)
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        tmp_path = os.path.join(tmp_dir, filename)
                        self.animator.create_animated_gif(tmp_path)
                        for logger in self.loggers.values():
                            logger.log_image_file(tmp_path, fig=fig, global_step=trainer.current_epoch)
                else:
                    for logger in self.loggers.values():
                        logger.log_figure(filename, fig, global_step=trainer.current_epoch)