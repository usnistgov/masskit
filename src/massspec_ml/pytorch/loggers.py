import warnings

from omegaconf import DictConfig, ListConfig
import tempfile
from omegaconf import OmegaConf
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers.mlflow import MLFlowLogger
from mlflow.utils.mlflow_tags import (
    MLFLOW_RUN_NAME,
    MLFLOW_SOURCE_NAME,
    MLFLOW_GIT_COMMIT,
    MLFLOW_USER
)
import logging
import sys
import os
import socket

try:
    import git
except ImportError:
    logging.info("Unable to import it")
    git = None

"""
mlflow loggers
"""


class MSMLFlowLogger(MLFlowLogger):
    def __init__(self, config, model, loader, artifacts, *args, **kwargs):
        """
        constructor for a logger for MLFlow. Shares function definitions with MSTensorBoardLogger

        :param config: configuration
        :param model: model
        :param loader: dataloader
        :param artifacts: directories to log
        """
        super().__init__(
            experiment_name=config.experiment_name,
            tracking_uri=config.logging.MSMLFlowLogger.tracking_uri,
            *args,
            **kwargs,
        )
        self.model = model
        self.loader = loader
        self.config = config
        self.mlf_setup_tags()
        self.log_params_to_mlflow()
        self.mlf_log_string(str(self.model), "model_summary.txt")

    def mlf_log_string(self, string_in, filename):
        """
        log omegaconf to mlflow

        :param string_in: string to write
        :param filename: name of the artifact file
        """
        with tempfile.TemporaryDirectory() as tempdir:
            with open(f"{tempdir}/{filename}", mode="w+") as fp:
                fp.write(string_in)
                fp.flush()
                self.experiment.log_artifact(run_id=self.run_id, local_path=fp.name)

    def log_params_to_mlflow(self):
        """
        log typical experiment parameters to mlflow
        """

        self.mlf_log_string(OmegaConf.to_yaml(self.config), "config.txt")
        self.log_params_from_omegaconf_dict(self.config.ms)
        self.log_params_from_omegaconf_dict(self.config.ml)
        self.log_params_from_omegaconf_dict(self.config.input)

    def log_params_from_omegaconf_dict(self, params):
        """
        recursively examine omegaconf object

        :param params: omegaconf object
        """
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element, enumerate_lists=False):
        """
        recursion function for log_params_from_omegaconf_dict

        :param parent_name: key of parent element
        :param element: value of parent element
        :param enumerate_lists: enumerate every element in a list
        """
        if isinstance(element, DictConfig):
            for k, v in element.items():
                if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                    self._explore_recursive(f"{parent_name}.{k}", v)
                else:
                    self.experiment.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif enumerate_lists and isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.experiment.log_param(self.run_id, f"{parent_name}.{i}", v)
        else:
            self.experiment.log_param(self.run_id, f"{parent_name}", element)

    def mlf_setup_tags(self):
        """
        create tags for logging as MLflow doesn't set these standard mlflow tags
        """
        self.mlf_set_tag(MLFLOW_RUN_NAME, None, self.run_id)
        logging.info(f'run id = {self.run_id}')
        self.mlf_set_tag(MLFLOW_USER, self.config.logging.user)
        self.mlf_set_tag(
            MLFLOW_SOURCE_NAME, self.config.logging.source.name, sys.argv[0]
        )
        if self.config.logging.git.commit:
            self.mlf_set_tag(MLFLOW_GIT_COMMIT, self.config.logging.git.commit)
        elif git:
            try:
                repo = git.Repo(search_parent_directories=True)
                self.mlf_set_tag(
                    MLFLOW_GIT_COMMIT,
                    self.config.logging.git.commit,
                    repo.head.object.hexsha,
                )
            except git.InvalidGitRepositoryError:
                pass
        self.mlf_set_tag("mlflow.note.content", self.config.logging.note.content)

        # log ip address and hostname
        hostname = socket.gethostname()
        self.mlf_set_tag("hostname", hostname)
        try:
            ip = socket.gethostbyname(hostname)
            self.mlf_set_tag("ip", ip)
        except Exception as e:
            pass

        # model information tag is MLFLOW_LOGGED_MODELS, but needs to be of the form
        # https://github.com/mlflow/mlflow/blob/1790dfaa01ca51dc200b1d2bff66162d90abbff8/mlflow/server/js/src/common/utils/Utils.js#L592
        # list of
        # {'artifact_path': 'best_model',
        #  'flavors': {'keras': {'data': 'data',
        #                        'keras_module': 'tensorflow.keras',
        #                        'keras_version': '2.3.0-tf'},
        #              'python_function': {'data': 'data',
        #                                  'env': 'conda.yaml',
        #                                  'loader_module': 'mlflow.keras',
        #                                  'python_version': '3.6.9'}},
        #  'run_id': '43d7deafb5b74cf489f2ecbbc9e3828c',
        #  'utc_time_created': '2020-07-03 06:31:25.337757'}
        # where utc_time_created, artifact_path, flavors.python_function are required

    def mlf_set_tag(self, tag, config_value, process_value=None):
        """
        set a particular tag with the config value.  if null config value, then use process value

        :param tag: the tag
        :param config_value: the configuration value
        :param process_value: the process value
        :return:
        """
        if config_value:
            self.experiment.set_tag(self.run_id, tag, config_value)
        elif process_value:
            self.experiment.set_tag(self.run_id, tag, process_value)

    def log_figure(self, figure_tag, fig, global_step=None):
        """
        log a matplotlib figure as an artifact

        :param figure_tag: name of the image
        :param fig: the matplotlib figure
        :param global_step: the epoch
        """
        self.experiment.log_figure(self.run_id, fig, figure_tag)

    def log_image_file(self, filename, fig=None, global_step=None):
        """
        log an image file, including animated gifs, or, if the logger does not support saving files,
        log an image of the matplotlib figure

        :param fig: matplotlib figure
        :param filename: what to name the image
        :param global_step: epoch
        """
        self.experiment.log_artifact(self.run_id, filename)

    def close(self, artifacts=None, *args, **kwargs):
        """
        close the log, saving tensorboard artifacts if available to mlflow

        :param artifacts: directory with artifact directries to log
        """
        if artifacts:
            for artifact_name, artifact in artifacts.items():
                self.experiment.log_artifacts(
                    self.run_id, artifact, artifact_path=artifact_name
                )


class MSTensorBoardLogger(TensorBoardLogger):
    def __init__(self, config, model, loader, artifacts, *args, **kwargs):
        """
        constructor for a logger for TensorBoard. Shares function definitions with MSMLFlowLogger

        :param config: configuration
        :param model: model
        :param loader: dataloader used to initial model so a graph can be created.
        :param artifacts: directories to log
        """
        super().__init__(
            config.logging.MSTensorBoardLogger.log_dir,
            name=config.experiment_name,
            *args,
            **kwargs,
        )
        self.model = model
        self.loader = loader
        self.config = config
        artifacts["events"] = config.logging.MSTensorBoardLogger.log_dir
        self.log_hyperparams(self.config.ml)
        self.experiment.add_text("config.yaml", OmegaConf.to_yaml(self.config))
        self.experiment.add_text("model_summary", str(self.model))

    def log_figure(self, figure_tag, fig, global_step=None):
        """
        log a matplotlib figure as an artifact

        :param figure_tag: name of the image
        :param fig: the matplotlib figure
        :param global_step: the epoch
        """
        self.experiment.add_figure(figure_tag, fig, global_step=global_step)

    def log_image_file(self, filename, fig=None, global_step=None):
        """
        log a list of images as an animated gif, or, if the logger does not support animated gifs,
        log an image of the matplotlib figure

        :param fig: matplotlib figure
        :param filename: what to name the image
        :param global_step: epoch
        """
        if fig:
            self.log_figure(os.path.basename(filename), fig, global_step=global_step)

    def close(self, *args, **kwargs):
        """
        close the logger, adding graph to the log.  currently disabled as there are multiple problems with
        using torch.jit.trace.

        :return:
        """
        return
        # batch = next(iter(self.loader.create_loader("valid")))
        # self.model.eval()
        # for non-deterministic model, will give a warning "TracerWarning: Output nr 1. of the traced function..."
        # as torch.jit.trace is called without setting check_trace=False
        # for non spectrum model, complains about missing attribute for bins property.
        # self.experiment.add_graph(self.model, (batch,))
        # self.model.train()


def filter_pytorch_lightning_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The dataloader.*does not have many workers",
        UserWarning,
        "pytorch_lightning",
    )
    warnings.filterwarnings(
        "ignore",
        r"No correct seed found, seed set to",
        UserWarning,
        "pytorch_lightning",
    )