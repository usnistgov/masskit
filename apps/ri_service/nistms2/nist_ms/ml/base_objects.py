from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np


def get_model_by_name(model_module_in, model_name, config_in):
    """
    method for generic retrieval of ChemModel by class name within a file
    :param model_module_in: the name of the module containing the model class
    :param model_name: the name of the model
    :param config_in: standard config object
    :return: keras model
    """
    model_class = getattr(model_module_in, model_name)
    model_object = model_class(config_in)
    last, penultimate, inputs = model_object.get_model()
    return Model(inputs, [last])


class ChemModel:
    """
    base class for a trainable chemistry model
    """
    def __init__(self, config, input_shape=None, final_layer_size=None, activation_name=None):
        """
        initialize object.  note that some params are for convenience.  All params are used from config
        :param config: the standard configuration object, subclassed from ModelParams
        :param input_shape: the numpy shape of the input
        :param final_layer_size: number of neurons in the final layer
        :param activation_name: what activation to use, e.g. softmax or linear
        """
        if input_shape is not None:
            config.input_shape = input_shape
        if final_layer_size is not None:
            config.final_layer_size = final_layer_size
        if activation_name is not None:
            config.activation_name = activation_name
        self.config = config
        return

    def get_model(self, name=None):
        """
        create the model
        :param name: name of model
        :return: last layer, penultimate layer, input layer
        """
        return None, None, None


class ChemDataGenerator(Sequence):
    """
    Generates chemical data for Keras
    """
    def __init__(self, list_ids, config_in, df_in, shuffle=False, augment=False):
        """
        Initialize
        :param list_ids: list of records ids
        :param config_in: configuration
        :param df_in: dataframe containing list of records
        :param shuffle: should the records be shuffled?
        :param augment: turn on augmentation
        """

        if config_in.batch_size % config_in.num_gpus != 0:
            raise ValueError(f"The batch size of {config_in.batch_size} is not an integer multiple"
                             f" of the number of gpus, {config_in.num_gpus}.")

        self.config = config_in
        self.list_ids = list_ids
        self.shuffle = shuffle
        self.df = df_in
        self.indexes = None  # indexes into list_ids.  Inited below in call to on_epoch_end()
        self.augment = augment
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: the number of batches in an epoch
        """
        # get number of full batches
        full_batches = int(np.floor(len(self.indexes) / self.config.batch_size))
        # get size of last batch
        last_batch_size = len(self.indexes) - full_batches * self.config.batch_size
        # if last batch is smaller than number of gpus, return full batch size
        if last_batch_size < self.config.num_gpus:
            return full_batches
        # otherwise, return full batch size plus one
        return full_batches + 1

    def __getitem__(self, index):
        """
        Generate one batch of data.  size of batch will be a nonzero multiple of number of gpus,
        which is required in keras/tensorflow to use multiple gpus.
        :param index: position of batch in indexes
        :return: (input, output)
        """
        # Generate indexes of the batch
        indexes = self.indexes[index*self.config.batch_size:(index+1)*self.config.batch_size]
        # divide by number of gpus
        quotient, remainder = divmod(len(indexes), self.config.num_gpus)
        # make batch size an integer multiple of num_gpus
        indexes = indexes[0:quotient*self.config.num_gpus]
        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x_in, y = self.__data_generation(list_ids_temp)

        return x_in, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """

        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            if self.config.minibatch_conformer:
                raise ValueError("you cannot shuffle when using minibatches to hold all conformers of a molecule")
                # this is because shuffling would cause the identical mol ids to be not in the same minibatch
        return

    def __data_generation(self, list_ids_temp):
        """
        calls data_generation to allow for overriding of method.  Functions that start with double underscore
        are not overridable in python.
        :param list_ids_temp: list of record ids
        :return: processed (input, output)
        """
        return self.data_generation_x(list_ids_temp), self.data_generation_y(list_ids_temp)

    def data_generation_x(self, list_ids_temp):
        """
        Generates input (x) data containing batch_size samples
        :param list_ids_temp: list of record ids
        :return: processed input
        """
        pass

    def data_generation_y(self, list_ids_temp):
        """
        Generates output data (y) containing batch_size samples
        :param list_ids_temp: list of record ids
        :return: processed output
        """
        pass


class SpectraDataGenerator(ChemDataGenerator):
    """
    Generates spectral data for Keras
    modification of ChemDataGenerator that allows for a query and a hit datafram
    """
    def __init__(self, list_ids, config_in, df_query_in, df_search_in, shuffle=False, augment=False):
        """
        Initialize
        :param list_ids: list of records ids
        :param config_in: configuration
        :param df_query_in: dataframe containing list of records queried
        :param df_search_in: dataframe containing list of records searched
        :param shuffle: should the records be shuffled?
        :param augment: turn on augmentation
        """
        self.df_search = df_search_in
        super(SpectraDataGenerator, self).__init__(list_ids, config_in, df_query_in, shuffle, augment)


def create_callbacks(output, histogram_freq=0, write_images=False, write_grads=False, no_checkpoints=False):
    """
    create tensorflow callbacks for training
    :param output: output name for models
    :param histogram_freq: what is the frequency of histogram writing
    :param write_images: should images be written?
    :param write_grads: should gradients be written?
    :param no_checkpoints: don't create callbacks
    :return: callback list
    """
    callbacks_list = []
    if not no_checkpoints:
        # setting write_grads to True can cause error on second epoch
        callbacks_list = [
            ModelCheckpoint("%s.{epoch:02d}-{val_loss:.6f}.h5" % output, monitor='val_acc'),
            TensorBoard(log_dir="logs", histogram_freq=histogram_freq,
                        write_images=write_images, write_grads=write_grads)
        ]
    return callbacks_list