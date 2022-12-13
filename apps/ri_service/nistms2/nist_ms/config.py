import platform
import numpy as np
# import git
import logging


class BaseConfig:
    pass


class BaseMLConfig(BaseConfig):
    """
    Parameters shared by all machine learning configurations
    """
    learning_rate = 1e-4
    momentum = 0.99
    learning_rate_step = 0.01  # how much to reduce the learning rate per set/step of epochs
    augment_rotate = False  # if image/tensor input, randomly rotate
    augment_flip = False  # flip augmentation
    random_conformer = False  # randomly pick a conformer
    num_gpus = 1  # the number of gpus.  batch size must be a multiple of this.
    truncate = 0  # number of records to truncate training/test set to.  0=none
    optimizer = "SGD"  # the optimizer to use
    epochs = 10  # number of epochs
    batch_size = 2  # number of molecules per batch
    if platform.system() == "Windows":
        data_path = "\\Users\\lyg\\data\\deep\\spectra\\ei"
    elif platform.system() == "Linux":
        data_path = "/wrk/lyg/projects2/data/deep/spectra/ei"


class ModelParams:
    """
    parameters specific to neural net models
    """
    skip_block_5 = True  # for VGG, skip block 5
    block_1_filters = 64  # for VGG and resnet, number of filters for first layer
    block_1_conv_size = 3  # for VGG, size of convolution in first layer.  use 7 for resnet
    skip_start = False  # for densenet, whether to skip first transition layer
    compression_factor = 0.5  # for densenet, what compression factor to use
    growth_rate = 32  # for densenet, the growth rate
    start_channels = 64  # for densenet, the number of filter channels in the first transition layer
    nalu_size = 0  # use nalu for final layers. size is the size of the second to last nalu layer. 0 = don't use nalu
    dropout = 0.0  # the amount of dropout to use
    kernel_regularizer = "None"  # the regularization function to use, e.g. regularizers.l2(0.01)
    input_shape = None  # the shape of the input
    activation_name = None  # what final activation to use: linear or softmax
    final_layer_size = 1  # the size of the final layer
    # if not zero, create an extra dense layer with size of this parameter.  currently only for resnet50_3d
    extra_dense_layer = 0
    output_layer_name = "final"
    input_layer_name = "input"
    short_model = False  # for resnet2, truncate the network to two layers

    class simple:  # this is for a model that is in development
        num_intermediate = 2
        intermediate_size = 1000
        batch_norm = False
        kernel_regularizer = None

    class seq:  # transformer model
        d_model = 128
        num_layers = 6
        num_heads = 32
        dff = 4096

    class attention:  # attention based model
        num_layers = 1

    class one_d:  # one dimensional spectral model
        # how many times to multiply the input channel size for the channel size of the first layer
        initial_channel_multiplier = 2
        # how many times to multiply the number of subsequent layer channels
        subsequent_channel_multiplier = 2

# todo: tease apart small molecule configuration from machine learning on small molecule configuration
# for example, model_3d setting should be in the machine learning config, not the small molecule config


class SmallMolMLConfig(BaseMLConfig):
    """
    configuration for small molecule machine learning
    """
    # allowable atomic numbers.  0=no atom
    # none, H, ...
    # I deleted duplicate 27.  Was it a typo?
    element_pos = 1  # position of the elements in encoding
    atomic_nums  = np.array([ 0, 1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 26, 27, 34, 35, 50, 53], dtype="int32")
    atomic_group  = np.array([0, 1,13,14,15,16,17, 14, 15, 16, 17,  8,  9, 16, 17, 14, 17], dtype="int32")
    atomic_period = np.array([1, 2, 2, 2, 2, 2, 2,  3,  3,  3,  3,  4,  4,  4,  4,  5,  5], dtype="int32")
    partial_charge_norm = 2.0   # value used to norm partial_charge partial charges
    partial_charge = False  # calculate partial_charge partial charges for embedding?
    partial_charge_pos = 11  # position of the partial charges in the encoding
    bonds = True  # use bonds in embedding
    abbreviated = True  # use a compressed feature vector for atoms
    abbreviated_bonds = True  # use a compressed feature vector for bonds
    implicit_hydrogens = False  # don't make hydrogens explicit
    full_bonds = False  # draw bonds from starting to ending atoms
    sasa = False  # calculate solvent accessible surface area for each atom
    sasa_norm = 256.0  # normalization factor for solvent accessible surface area
    sasa_pos = 12  # position of sasa in the encoding
    probeRadius = 1.4  # size of the sasa probe radius
    bond_length = False  # encode the length of the bond
    bond_length_norm = 3.0  # normalization factor for bond length
    bond_length_pos = 13  # position of the bond length in the encoding
    vdw_radius = False  # for an atom, encode the van der Waals radius
    vdw_radius_norm = 3.5  # normalization factor for vdw radius
    vdw_radius_pos = 14  # position of the vdw radius in the encoding
    bond_dipole = False  # calculate the dipole of two bonded atoms
    bond_dipole_norm = 6.0  # normalization factor for the dipole of two bonded atoms
    bond_dipole_pos = 15  # position of the bond dipole
    polarizability = False  # calculation per atom polarizability
    polarizability_pos = 16  # position of the polarizability feature
    polarizability_norm = 1000  # used to norm the polarizability
    xyz = False  # include the xyz coordinates (currently only implemented in sequence)
    xyz_pos = 17  # position of the x y z coordinates
    cube_dim = 64  # size of cube to hold molecule
    cube_size = 16.32  # side of cube in angstroms (0.255 angstroms cubed voxels)
    max_bound = 8.16  # the maximum extent along any axis of the 3d conformer

    # fingerprint sizes
    fingerprint_size = 167  # size of rdkit MACCS fingerprint
    nist_fingerprint_size = 772  # size of NIST fingerprint
    ecfp2_fingerprint_size = 1024  # size of ecfp2 fingerprint
    ecfp4_fingerprint_size = 1024  # size of ecfp4 fingerprint

    # enumeration of fingerprint bit sets
    full_fingerprint = [x for x in range(1, fingerprint_size)]  # the entire MACCS fingerprint
    nist_fingerprint = [x for x in range(1, nist_fingerprint_size)]  # the entire nist fingerprint
    ecfp2 = [x for x in range(0, ecfp2_fingerprint_size)]
    ecfp4 = [x for x in range(0, ecfp4_fingerprint_size)]
    simple_fingerprint = [161, 164]  # N and O fingerprint
    simple_groups = [11, 13, 15, 19, 21, 22, 23, 24, 25, 30, 34, 37, 38, 41, 43, 45, 48, 50, 52, 53, 54,
                     56, 63,
                     65, 66, 68, 69, 70, 71, 72, 74, 76, 77, 78, 79, 80, 82, 84, 85, 89, 92, 93, 94, 95,
                     96, 97, 99, 100, 102,
                     104, 108, 109, 110, 112, 113, 114, 115, 116, 117, 119, 122, 123, 124, 132, 135, 139, 144,
                     148, 151, 152, 153, 154, 156, 157, 158, 160, 161, 162, 164]  # standard fingerprint groups

    # list of fingerprint bit sets (from above)
    fingerprints = ["full_fingerprint", "simple_fingerprint", "simple_groups", "nist_fingerprint", 'ecfp2', 'ecfp4']
    # which column in the pandas dataframe to use to retrieve the fingerprint
    fingerprint_column = {"full_fingerprint": "fingerprint", "simple_fingerprint": "fingerprint",
                          "simple_groups": "fingerprint", "nist_fingerprint": "nist_fingerprint", 'ecfp2': 'ecfp2',
                          'ecfp4': 'ecfp4'}
    # the string name of the fingerprint in use
    fingerprint = "full_fingerprint"
    # a pointer to the fingerprint bit set in use
    fingerprint_bits = full_fingerprint  # which fingerprint bits to use.  0 is unused
    ri_field = "experimental_ri"  # which field to use for ri: experimental_ri or estimated_ri
    model_type = "ri"  # what are we predicting? ri, fingerprint, usr
    dim = "3d"  # is it a 2D or 3D model?
    usr_norm = 20.0  # value to norm the USR values
    minibatch_conformer = False  # use the minibatch to hold 3d conformers of a single molecule
    ignore_undef = False  # filter out records with undefined stereocenters
    model_2d = "vgg16_2d"  # the name of the 2d model to use
    model_3d = "vgg16_3d"  # the name of the 3d model to use
    zero_extra_channels = False  # zero out partial_charge and sasa channels
    max_atoms_bonds = 50  # the maximum number of atoms and bonds for sequence models
    heavy_cutoff = 0  # the maximum number of heavy atoms  (0 = no cutoff)
    atom_one_hot_len = len(atomic_nums)  # one hot encoding length for atoms
    hydrogen_len = 4  # max number of hydrogens per atom

    # feature length for an atom
    atom_feature_len = atom_one_hot_len + hydrogen_len

    # feature length for stereo
    # Chem.AssignBondStereoCodes(rdmol); rdmol.GetBondWithIdx(1).GetStereo()
    # Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    # m.getAtomWithIdx(i).getChiralTag() != Atom.ChiralType.CHI_UNSPECIFIED
    # m.getAtomWithIdx(i).hasProp("_ChiralityPossible")
    # values are_ChiralityPossible + CHI_UNSPECIFIED, CHI_OTHER, CHI_TETRAHEDRAL_CW, CHI_TETRAHEDRAL_CC,
    # STEREONONE, STEREOANY, STEREOZ, STEREOE, STEREOCIS, STEREOTRANS
    # STEREOANY means a crossed bond, similar idea to _ChiralityPossible + CHI_UNSPECIFIED
    stereo_feature_len = 10

    # total feature length.  last two positions are for the distance and topological matrix entries
    total_feature_len = atom_feature_len * 2 + stereo_feature_len + 2

    boost_score = 0.1  # how much to boost the score of an exact match
    tanimoto_clip = 0.0  # if nonzero, clip the lower bound of the tanimoto score


class BaseMSConfig(BaseConfig):
    """
    Parameters shared by all mass spec configurations
    """

# todo: tease apart EI spectra config from ri config.


class EIConfig(BaseMSConfig):
    """
    configuration for gc/ms with ei
    """
    max_mz = 2000.0  # the maximum mz value
    max_intensity = 1000.0  # the maximum intensity value
    product_precision = 1.0
    width_spectrum = int(max_mz / product_precision)
    max_ri = 6362.0  # maximum retention index (for normalization)
    ri_cutoff = 4200  # maximum ri value for training


class TandemConfig(BaseMSConfig):
    """
    configuration for tandem spectra
    """
    max_mz = 1000.0
    max_intensity = 1000.0
    product_precision = 10.0/1000000  # ppm precision
    max_energy = 200.0  # used to normalized the energy


class TandemMLConfig(TandemConfig, ModelParams, SmallMolMLConfig):
    """
    config for 1d tandem spectra training
    """
    model = "resnet50_1d_hires_large"
    boost_score = 0.2  # how much to boost the score of an exact match
    tanimoto_clip = 0.0  # if nonzero, clip the lower bound of the tanimoto score
    # bin_size = 100 * TandemConfig.product_precision  # precision at 100 Da. give about 1M bins in a fixed array
    bin_size = 0.01
    augment_noise = False  # augment query spectra with noise
    num_search_results = 100000   # number of search results to use for training
    energy_channel = False  # create channel in the spectra that encodes the energy
    mz_channel = False  # create channel containing mz.  normalized by max_mz
    precursor_minus_mz_channel = False  # create channel containing precursor-mz.  normalized by max_mz
    num_beginning_layers = 2

    def get_input_shape(self, num_spectra=1):
        """
        compute the shape of the input tensor for a spectrum
        :param num_spectra: number of spectra per input tensor
        """
        num_bins = int(self.max_mz/self.bin_size)
        num_channels = 1 + self.energy_channel + self.mz_channel + self.precursor_minus_mz_channel
        return num_bins, num_channels * num_spectra


class EIMLConfig(EIConfig, SmallMolMLConfig, ModelParams):
    """
    config for machine learning on EI data
    """
    img_size = 299  # size of image
    loss = "binary_crossentropy"  # the loss function to use
    model = "resnet50_1d"
    bin_size = 1  # size of bin in fixed spectra array

    @classmethod
    def get_input_shape(cls, num_spectra=1):
        """
        compute the shape of the input tensor for a spectrum
        :param num_spectra: number of spectra per input tensor
        """
        num_bins = cls.max_mz/cls.bin_size
        num_channels = 1
        return num_bins, num_channels * num_spectra


def log_param_git(config: BaseConfig):
    """
    function to log config parameters and the git sha
    :param config: the configuration to print out
    """
    # try:
    #    with git.Repo(search_parent_directories=False) as repo:
    #        sha = repo.head.object.hexsha
    #        logging.info(f"git sha is {sha}")
    # except:
    #    pass
    logging.info('\n'.join("%s: %s" % item for item in vars(config).items()))
    return
