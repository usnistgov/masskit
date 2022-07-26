# Peptide Prediction Models
## Setup for training peptide prediction models

### Initial setup
#### Local linux machine setup
* [install anaconda](https://www.anaconda.com/products/individual) if not already installed: 
  * `mkdir ~/miniconda3`
  * `curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --output ~/miniconda3/miniconda.sh`
  * `bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3`
  * `~/miniconda3/bin/conda init bash`
  * edit `~/.bash_profile` to include the lines `source ~/.bashrc` and `conda activate msml`, then log out and log back in.
* then download and install the software libraries
  * `mkdir ~/source; cd ~/source; git clone https://gitlab.nist.gov/gitlab/msdc/msdc_services.git`
    * you may have to ask [Douglas](mailto:douglas.slotta@nist.gov) or [Lewis](mailto:lewis.geer@nist.gov) or for access
  * `cd msdc_services`
  * `git checkout arrow`
  * `source environments/init_msml.sh`
  * `cd libraries`
  * `python setup.py develop`
  * `cd src/massspec/utils/`
  * `python setup.py build_ext --inplace`
* optionally, you can download the data.  If you are on AWS or a local machine, the data is downloaded and cached automatically in the first
  training run. However, if you start out with a multi job run, there can be a race condition during the download, so 
  manual installation is a good idea:
    * `mkdir -p  ~/data/nist/aiomics` to make the cache directory, then go to the directory `cd ~/data/nist/aiomics`
    * `curl -o ~/.massspec_cache/2021-05-25_00_HmConsSemi.db https://msdc-aiomics-data.s3.amazonaws.com/ChoLibCom20210701.parquet`
    * `curl -o ~/.massspec_cache/2021-05-25_00_HmSelLib.db https://msdc-aiomics-data.s3.amazonaws.com/ChoLibSim20210701.parquet`
    * `curl -o ~/.massspec_cache/2021-05-24_00_HmAll.db https://msdc-aiomics-data.s3.amazonaws.com/ChoLibUniq20210701.parquet`
    * `curl -o ~/.massspec_cache/2021-07-01_01_ChoLibUniq.db https://msdc-aiomics-data.s3.amazonaws.com/HmAll20210525.parquet`
    * `curl -o ~/.massspec_cache/2021-05-24_00_HmConsBest.db https://msdc-aiomics-data.s3.amazonaws.com/HmCons20210525.parquet`
    * `curl -o ~/.massspec_cache/2021-05-25_00_HmConsGood.db https://msdc-aiomics-data.s3.amazonaws.com/HmSelLib20210525.parquet`


#### Nisaba cluster setup  
[nisaba.nist.gov](https://gitlab.nist.gov/gitlab/HPC/dgx/-/wikis/home) is a compute cluster that runs job via the 
command line using the slurm scheduler.
* log into nisaba.nist.gov using your general realm username and your favorite ssh client, e.g. 
  `ssh lyg@nisaba.nist.gov`. Accounts can be requested via [nisaba-help@list.nist.gov](mailto:nisaba-help@list.nist.gov). 
* use the setup described above for local linux machines
* if you want to use jupyter notebooks on nisaba:
  * from the nisaba command line, install code to use jupyter notebooks: `conda install ipykernel`
  * type `python -m ipykernel install --user --name msml --display-name "msml"`
  * now, edit `~/.local/share/jupyter/kernels/msml/kernel.json` to add the line `,"env": {"PYTHONPATH": "/home/your_user_name/source/msdc_services/libraries/src"}` mafter the line `"language": "python"`.  Replace `your_user_name` with your unix account name.
  * log into the nisaba [jupyterhub](https://nisaba.nist.gov:8888/hub/login) from your browser. 
    * you may get an error message about untrusted certificates.  If you do, select the option to ignore the error and continue.
  * You can open a new notebook by clicking on the `+` sign at upper left, and then clicking on
    the `Python (msml)` button under `Notebook` in the `Launcher` tab.

### Running training 
#### Running training on a local linux machine
* `cd ~/source/msdc_services/apps/ml/peptide` or wherever you have cloned msdc_services
* `git pull origin master` to pull in any changes to the library
* configuration is managed by the hydra package. To configure, see [below](#configuration)
* run training by executing `python train.py`
    * to run on multiple gpus: `python train.py setup.gpus=2 setup.accelerator=ddp`
* if you are logging locally, examine the logs by first doing `cd hydra_output`
    * `mlflow ui`
    * `tensorboard --logdir tb_logs`
        * Please ignore the hp_metric as it's a dummy metric used to overcome a bug in tensorboard
        * text artifact display in tensorboard ignores linebreaks as tensorboard is expecting markdown
* if you are logging on the mlflow server, browser [here](http://10.208.85.216:5000).
    
#### Running training on nisaba and other slurm clusters
* Log into nisaba.nist.gov using your general realm username and your favorite ssh client, e.g. 
  `ssh lyg@nisaba.nist.gov`. Accounts can be requested via [nisaba-help@list.nist.gov](mailto:nisaba-help@list.nist.gov). 
* If you haven't done so already, follow the setup instructions [above](#initial-setup)
* cd to `~/source/msdc_services/apps/ml/peptide` or wherever you have cloned msdc_services
* `git pull origin master` to pull in any changes to the library
* to launch multiple jobs, create a shell script.
  * this will run train.py the same way as you run it from the command line, but as a batch job.  To do this, create a
    shell script called, for example, my_script.sh that looks something like:
    ```bash
    !/bin/bash
    #
    #SBATCH --time=72:00:00
    #SBATCH --partition=batch
    #SBATCH --gres=gpu:1
    #SBATCH --mem-per-cpu=10000
    #SBATCH -c 3
    
    source ~/.bashrc
    python train.py &> mylog.log
    ```
  * You can put anything you want into this shell script, including command line options to train.py.  Submit it as
    a job using `sbatch my_script.sh` and monitor it using `squeue`.  To kill it use `scancel 1234` where 1234 
    is the job number returned by sbatch.  mylog.log will have the same output as you see when running interactively
    and you can view it using `tail -f mylog.log`.
  * On nisaba, database files can be placed in a scratch directory on each node to maximize throughput.
    * First, copy the database files you want to use to the scratch directory for your slurm job by putting something like 
      `rsync /aiomics/massspec_cache/ChoLibUniq20210701.parquet /aiomics/massspec_cache/HmAll20210525.parquet /scratch/${SLURM_JOB_ID}` 
      at the start of shell script
    * Then, put the path to the copied databases in the command line parameters for train.py, e.g.
      `input.train.spectral_library=/scratch/${SLURM_JOB_ID}/HmAll20210525.parquet input.valid.spectral_library=/scratch/${SLURM_JOB_ID}/ChoLibUniq20210701.parquet`
    * The scratch directory for the slurm job will be deleted at the end of the job.
#### Output models
When using mlflow, each run will be given a run id.  On the mlflow server, all runs will be listed under the
experiment name and the best model will be uploaded as an artifact if the job terminates normally and 
is not killed.  Also, a copy of the current best model for each run will be saved to disk in a 
subdirectory of the "best_model" directory named after the run id.  The saving to disk happens
after every epoch, so this mechanism doesn't require the job to terminate normally.


### Creating predictions
#### Creating predictions on a local linux machine
* `conda activate msml`
* `cd ~/source/msdc_services/apps/ml/peptide` or wherever you have cloned msdc_services
* `git pull origin master` to pull in any changes to the library
* prediction configuration is found in conf/config_predict.yaml.  
  * put the checkpoint of the model to use in prediction as an entry under `model_ensemble:`
  * additional checkpoints can be listed under `model_ensemble:` but they have to take the same input and output
    shape as the first model.
  * the number of draws per model is given by `model_draws:`.
  * if you are using dropout in prediction, set `dropout:` to True.
  * the output file name prefix is set by `output_prefix`.  File extensions will be added to the prefix.
* run training by executing `python predict.py`
  * use the form `python predict.py 'model_ensemble=["my_model.ckpt"]'` to specify the model you are using from the command line.
* output will be placed in the working directory, e.g. `hydra_output`

#### Creating predictions on nisaba
* log into nisaba
* `cd ~/source/msdc_services/apps/ml/peptide` or wherever you have cloned msdc_services
* `git pull origin master` to pull in any changes to the library
* prediction configuration is found in conf/config_predict.yaml.  
  * put the checkpoint of the model to use in prediction as an entry under `model_ensemble:`
  * additional checkpoints can be listed under `model_ensemble:` but they have to take the same input and output
    shape as the first model.
  * the number of draws per model is given by `model_draws:`.
  * if you are using dropout in prediction, set `dropout:` to True.
  * the output file name prefix is set by `output_prefix`.  File extensions will be added to the prefix.
* create a shell script to run the prediction program
  * this will run predict.py the same way as you run it from the command line, but as a batch job.  To do this, create a
    shell script called, for example, my_predict_script.sh that looks something like:
    ```bash
    !/bin/bash
    #
    #SBATCH --time=72:00:00
    #SBATCH --partition=batch
    #SBATCH --gres=gpu:1
    #SBATCH --mem-per-cpu=3500
    #SBATCH -c 5
    
    source ~/.bashrc
    
    python predict.py &> my_predict_log.log
    ```
  * You can put anything you want into this shell script, including command line options to predict.py.  Submit it as
    a job using `sbatch my_predict_script.sh` and monitor it using `squeue`.  To kill it use `scancel 1234` where 1234 
    is the job number returned by sbatch.  my_predict_log.log will have the same output as you see when running interactively
    and you can view it using `tail -f my_predict_log.log`.

## Configuration
Configuration is handled by [hydra](https://hydra.cc). Hydra uses the human-readable yaml format as input and 
converts it into a dictionary-like object for use in python.  Using hydra organizes parameters,
simplifies changing groups of settings, automates the logging of parameters, and allows for hyperparameter sweeps.  
The hydra configuration can be found in [apps/ml/peptide/conf](apps/ml/peptide/conf) and its subdirectories.  The subdirectories,
which contain yaml files, are used to organize the configuration into logical submodules.  

The top level configuration file is [apps/ml/peptide/conf/config.yaml](apps/ml/peptide/conf/config.yaml).  It has a `defaults` section that
includes yaml files from subdirectories:
```yaml
defaults:
  - input: 2021-04-20_nist  # input data 
  - setup: single_gpu  # experiment setup parameters that are not typical hyperparameters
  - logging: mlflow_server  # logging setup. mlflow_server will log to the mlflow server.  
  - ml/model: AIomicsModel  # model parameters
  - ml/embedding: peptide_basic_mods  # embedding to use
  - ms: tandem  # mass spec parameters
```
The keys to the left are both keys in the top level of the configuration and the names of subdirectories under
conf/.  The values to the right are the yaml file names without the `.yaml` extension. These file names are NOT keys in
the configuration.  So to swap out portions of the config, you just specify a different file name.
For example, to first run the AIomicsModel on tandem spectra and then run it on EI spectra is
just a process of changing two file names:
```yaml
defaults:
  - input: 2021-01-01_nist_ei  # input data 
  - setup: single_gpu  # experiment setup parameters that are not typical hyperparameters
  - logging: mlflow_server  # logging setup
  - ml/model: AIomicsModel  # model parameters
  - ml/embedding: peptide_basic_mods  # embedding to use
  - ms: ei  # mass spec parameters
```

Note that as discussed above, the value of ml/model is a filename and is not required to be the name of the model 
class, which is given in the configuration file itself. This gives you the ability to have multiple configuration files
for the same model.

Hydra settings can be overridden from the command line.  For example, `python train.py ms.bin_size=1 ml.max_epochs=100`.

Hyperparameter sweeps can be specified from the command line also: `python train.py --multirun --ms.bin_size=0.1,1`

## Using checkpoints and transfer learning
### Checkpoints
The library will automatically save the last k best models as determined by the validation loss, where 
k is set by logging.save_top_k.  These k models are currently saved to the filesystem.  At the end of the training
epochs, the very best model is logged to mlflow.
Checkpoint files contain all information to restart training, including the configuration.
To restart training from a checkpoint, set input.checkpoint_in to the name of the checkpoint file.  Pytorch
lightning insists on putting = signs into the checkpoint filename and these = signs can be escaped by 
placing a backslash in front of the filename. ml.transfer_learning should be set to false.
### Transfer learning
Transfer learning uses the same settings as loading in checkpoints, but with ml.transfer_learning set to
True.  When this is set, the configuration setting in the checkpoint file are ignored and replaced with 
the current configuration settings.
### Bayesian networks
Bayesian layers are turned on by setting ml.bayes to True.  The number of samples take per batch is set by
ml.bayesian_network.sample_nbr.

## Creating models, losses, and new inputs or outputs to models
### Software architecture
* External libraries
  * [pytorch](https://pytorch.org/)
  * [bayesian-torch](https://github.com/IntelLabs/bayesian-torch): bayesian layers for pytorch
  * [pytorch lightning](https://www.pytorchlightning.ai/): used to organize pytorch code and to assist in logging and parallel training
  * [mlflow](https://www.mlflow.org/): experimental logging
  * [pyarrow](https://arrow.apache.org/docs/python/index.html): data storage and data structure specification
  * [pandas](https://pandas.pydata.org/): in memory storage of above data
  * [hydra](https://hydra.cc/): configuration management
  * [rdkit](https://www.rdkit.org/): small molecule cheminformatics
* MSDC Libraries
  * [massspec](libraries/src/massspec): manipulation of mass spectra
  * [massspec_ml/pytorch](libraries/src/massspec_ml/pytorch): mass spectral machine learning code
* Architecture
  * pytorch_lightning.Trainer [apps/ml/peptide/train.py](apps/ml/peptide/train.py): overall driver of training process.
    * SpectrumLightningModule [massspec_ml/pytorch/spectrum/spectrum_lightning.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_lightning.py) contains the train/valid/test
      loops.  Derived from pytorch_lightning.LightningModule, which in turn is derived from torch.nn.Module.
      * config (also hparams): configuration dictionary of type hydra.DictConfig.
      * model: the model being trained. SpectrumModel [massspec_ml/pytorch/spectrum/spectrum_base_objects.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_base_objects.py) derived
        from torch.nn.Module.
        * input and output are namedtuples that allow for adding multiple inputs and outputs to the model.
        * configured using the hydra.DictConfig config.
      * loss_function: loss function derived from BaseLoss [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py]),
        which is derived from torch.nn.Module. Takes the same namedtuples that are the input and output of the model.
    * SpectrumDataModule [massspec_ml/pytorch/spectrum/spectrum_lightning.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_lightning.py) derived from 
      pytorch_lightning.LightningDataModule.
      * creates TandemArrowDataset data loader [massspec_ml/pytorch/spectrum/spectrum_datasets.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_datasets.py) derived from BaseDataset,
        which in turn is derived from torch.utils.data.DataLoader.
        * embedding: input embeddings calculated by EmbedPeptide [massspec_ml/pytorch/spectrum/peptide/peptide_embed.py](libraries/src/massspec_ml/pytorch/spectrum/peptide/peptide_embed.py)
        * store: dataframes are managed by ArrowLibraryMap (massspec/utils/index.py) and its base classes.
          * integration with pandas dataframes provided by accessors defined in massspec/data_specs/spectral_library.py.
    * PeptideCB [massspec_ml/pytorch/spectrum/peptide/peptide_callbacks.py](libraries/src/massspec_ml/pytorch/spectrum/peptide/peptide_callbacks.py): logging at the end of validation epoch. Derived from
      pytorch_lightning.callbacks.base.Callback.
    * MSMLFlowLogger and MSTensorBoardLogger loggers [massspec_ml/pytorch/loggers.py](libraries/src/massspec_ml/pytorch/loggers.py).
### Custom models
- Models are standard `torch.nn.Module`'s and derived from `SpectrumModel` in [massspec_ml/pytorch/spectrum/spectrum_base_objects.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_base_objects.py)
  - modules within the models are derived from `SpectrumModule` in [massspec_ml/pytorch/spectrum/spectrum_base_objects.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_base_objects.py)
  - both `SpectrumModel` and `SpectrumModule` should be initialized with the config dictionary and should
    pass the config dictionary into the initializer for their superclasses.  
- To create a new model, subclass it from `SpectrumModel` and put it in an existing file in 
  [massspec_ml/pytorch/spectrum/peptide/models](libraries/src/massspec_ml/pytorch/spectrum/peptide/models) or create a new file.  Let's call the new model `MyModel`. See 
  [massspec_ml/pytorch/spectrum/peptide/models/dense.py](libraries/src/massspec_ml/pytorch/spectrum/peptide/models/dense.py) for a simple example of a model.
  - if you created a new file to hold the code for the model, append the filename to the configuration setting
    `modules.models` in [apps/ml/peptide/conf/paths/standard.yaml](apps/ml/peptide/conf/paths/standard.yaml)
- Configuration
  - `SpectrumModel` contains a `self.config` object.  This object is a dictionary-like object created
    from the yaml files under [apps/ml/peptide/conf](apps/ml/peptide/conf) along with any command line settings.  By using this config
    object, your parameters will automatically be logged and you can do automated sweeps.
  - use `self.config` to hold your configuration values.  To create the configuration for `MyModel`:
    - create a yaml file called `MyModel.yaml` in the directory [apps/ml/peptide/conf/ml/model](apps/ml/peptide/conf/ml/model) with your configuration values in
      it.  Use [apps/ml/peptide/conf/ml/model/DenseSpectrumNet.yaml](apps/ml/peptide/conf/ml/model/DenseSpectrumNet.yaml) as an example.
    - the top node of the configuration should be the name of the new class: `MyModel:`
    - make sure `# @package _group_` is the first line of the file.
    - then add configuration values to `MyMode.yaml` indented underneath `MyModel:`, such as `my_config: 123`.  You can then 
      reference it in the code for `MyModel` as `self.config.ml.model.MyModel.my_config`.
  - to use the model for training, edit [apps/ml/peptide/conf/config.yaml](apps/ml/peptide/conf/config.yaml) by
    - changing the line that starts with `ml/model` in `defaults:` to have the value `MyModel`, that is, the name of the
      new configuration file.
    - if you've created a new file for the model itself, then add the name of this new file to the list under `module.models:`
      in `config.yaml`.  This will tell the training program where to look for new models.
- Model input
  - the standard input to a model is a `namedtuple` called `ModelInput` defined in [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py)
    It has 3 elements:
    - `x`: the input tensor of shape `(batch, self.channels, self.config.ml.embedding.max_len)` where `self.channels` is
      the size of the embedding and `self.config.ml.embedding.max_len` the maximum peptide length
    - `y`: the experimental spectra of shape `(batch, ?, self.bins)`, where the second dimension are channels, usually one
      for intensity, and `self.bins` is the number of mz bins.
    - `index`: the position of the corresponding data for the spectra in the input dataframe
  - the elements of `ModelInput` should be referred to by index, e.g. 0, 1, or 2, as tensorboard model graph logging won't work if 
    you refer to the elements by name.
  - `namedtuples` instead of dicts are used for input and output as there are some functions, like graph export,
    that won't work with dictionaries as they are not constant.  
- Model output
  - the standard output from a model is a `namedtuple` called `ModelOutput` defined in [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py)
    It has 2 elements:
    - `y_prime`: the predicted spectra of shape `(batch, ?, self.bins)`, where the second dimension are channels, usually one
      for intensity, and `self.bins` is the number of mz bins.
      - channel 0 is intensity
      - an optional channel 1 is the standard deviation of the intensity
    - an optional `score` element used for scores created during the model forward(), such as KL divergence in Bayesian layers
- Bayesian networks
  - use `SpectrumModel` and `SpectrumModule` classes to construct your model.
  - to use Bayesian layers, set config.ml.bayesian_network.bayes to True.  Here is a 
    [list](https://github.com/IntelLabs/bayesian-torch/blob/main/doc/bayesian_torch.layers.md) of the layers
    that are available.
  - use the boolean config.ml.bayesian_network.bayes inside your model to turn on pytorch layers.
  - each layer will return two outputs, the normal tensor output and the kl divergence.  Sum the KL divergence
    across each boolean layer and pass it out of the model as the second argument of ModelOutput
  - use a loss that takes KL divergence, such as SpectrumCosineKLLoss or MSEKLLoss.
### Adding fields to the model input
* create your own version of `ModelInput`, lets call it `MyModelInput` and place it in [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py)
  * subclass `TandemArrowDataset` in [massspec_ml/pytorch/spectrum/spectrum_datasets.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_datasets.py) and place the new
    class, let's call it `MyDataSet`, in that file or another python file.
    * if you are creating a new python file, add it in [apps/ml/peptide/conf/paths/standard.yaml](apps/ml/peptide/conf/paths/standard.yaml) under `modules.dataloaders`
    * in the ms configuration you are using under `conf/ms`, set `dataloader:` to `MyDataSet`.
  * override `__get_item__` in `MyDataSet` to add the additional field and return it as a `MyModelInput`
    Note that `__get_item__ `only returns one row of information -- in later processing this data is batched and 
    moved into the GPU.  This later processing requires that the dictionary be flat and not nested, that is, each
    top level item in the dictionary should be vectorizable.
* alternatively, add fields to `ModelInput`
* data columns that may (or may not) be available from the dataframe are found in [massspec/data_specs/schemas.py](libraries/src/massspec/data_specs/schemas.py).
  * add any necessary data columns to `ms.columns` configuration you are using under `conf/ms`
### Adding fields to the model output
* create your own version of `ModelOutput`, let's call it `MyModelOuput` and place it in [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py)
  * modify your model to output `MyModelOuput`
* alternatively, add fields to `ModelOutput`.  Do not modify the numeric index of any field
### Custom losses
* losses are standard `torch.nn.Module`s and derived from `BaseLoss` in
  [massspec_ml/pytorch/base_objects.py](libraries/src/massspec_ml/pytorch/base_objects.py)
* The input to the losses are the `ModelInput` and `ModelOutput` as described above.  The reason
  to use these `namedtuples` is to give the loss functions access to all information fed to and returned
  by the models.
* `extract_spectra()` and `extract_variance()` are used to extract the intensity spectra and
  intensity variances from the input and output.
* to create your own loss:
  * subclass `BaseSpectrumLoss` or `BaseLoss` and place the loss, let's call it `MyLoss`,
    in [massspec_ml/pytorch/spectrum/spectrum_losses.py](libraries/src/massspec_ml/pytorch/spectrum/spectrum_losses.py) or place it in its own file.
  * if you created a new file to hold the code for the loss, append the filename to the configuration setting
    `modules.losses` in [apps/ml/peptide/conf/paths/standard.yaml](apps/ml/peptide/conf/paths/standard.yaml)
  * to use the loss, change `ml.loss.loss_function` in [apps/ml/peptide/conf/config.yaml](apps/ml/peptide/conf/config.yaml) to `MyLoss`.
### Custom metrics
* Metrics are measures of model performance that is not a loss, although a metric can use a loss function. To create a custom metric, start with the base classes in [massspec_ml/pytorch/metrics.py](libraries/src/massspec_ml/pytorch/metrics.py)
* from a loss
  * subclass `BaseLossMetric` to wrap an already existing loss specified by the parameter `loss_class`
* from scratch
  * subclass the new metric from `BaseMetric`. 
  * in the `__init__()`, use `self.add_state()` to add any tensors you need for the metric.  add_state() is used to initialize tensors as it will set them up so they can work across multiple gpus and multiple nodes. add_state() can also be used to initialize a list.
    create an `update()` function that takes the results of a minibatch and uses the results to update the tensors set up in __init__()
  * create a `compute()` function that takes the tensors and computes the metric.
* config
  * path to the metric modules is defined in `paths.modules.samplers`
  * to specify a sampler to use
    * during valid/test: `ml.valid_metrics`
    * during train: `ml.test_metrics`
    * during valid, test, and train: `ml.metrics`
### Custom sampler
* samplers allow weighted selection of input data to the network based on columns in the input data
* the base class `BaseSampler` for samplers is defined in [massspec_ml/pytorch/samplers.py](libraries/src/massspec_ml/pytorch/samplers.py)
  * to create a custom sampler, subclass `BaseSampler` and create a `probability()` method that computes an array where each element is the probability a corresponding row in the dataset is selected.
    * the probability does not have to be normalized
    * the fields available to probability() are the database fields listed in the configuration `ms.dataset_columns`, e.g. `self.dataset.data['peptide']`, `self.dataset.data['charge']`, `self.dataset.data['ev']`, `self.dataset.data['mod_names']`
* configuration
  * the path to sampler modules is defined in `paths.modules.samplers`
  * sampler to use is specified by `ml.sampler.sampler_type`.  Set this to `null` if no sampler should be used.
  * the data columns available to the sampler are specified in `ms.dataset_columns`.
  * configuration parameters for `LengthSampler`, which samples based on peptide length:
    * `max_length`: for this length of the peptide and longer, the probability of sampling is 1
    * `min_length`: for this length and smaller, the probability of sampling is `min_length*scale/max_length`
    * for lengths in between the probability of sampling linearly scales with length

## Miscellaneous settings
* Multiple validation files
  * Edit the input configuration file, e.g. `2021-05-24_nist.yaml` so that valid.spectral_library is a list of 
    validation libraries.  For example,
    ```yaml
    valid:
      spectral_library:
        - https://msdc-aiomics-data.s3.amazonaws.com/ChoLibSim20210701.parquet
        - https://msdc-aiomics-data.s3.amazonaws.com/ChoLibUniq20210701.parquet
    ```
* To log record ids used for each training epoch:
  * set `input.train.log_ids` to `True`
  * the files containing the ids for each epoch will be found in the working directory with
    filenames of form `log_ids_epoch_*.txt`
  * use of this option will slow down training.


  
