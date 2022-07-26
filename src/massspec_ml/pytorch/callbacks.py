import pytorch_lightning as pl
import os
import glob


class ConcatenateIdLogs(pl.callbacks.base.Callback):
    """
    used to concatenate log files created from each training worker thread
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        with open(f"log_ids_epoch_{trainer.current_epoch}.txt", 'w') as outfile:
            for filename in glob.glob('log_worker_*.txt'):
                with open(filename) as infile:
                    for line in infile:
                        outfile.write(line)
                os.remove(filename)


class ModelCheckpointOnStart(pl.callbacks.ModelCheckpoint):
    """
    used to save parameters before training begins
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def on_train_start(self, trainer, pl_module):
        if 'save_untrained' in self.config.logging and self.config.logging.save_untrained:
            self.save_checkpoint(trainer, pl_module)


