import pyarrow as pa


def name_to_type(schema, type_name):
    """
    convert field name to arrow type of field

    :param schema: schema to search
    :param type_name: name of type
    :return: type
    """
    index = schema.get_field_index(type_name)
    if index == -1:
        raise ValueError(f'type {type_name} not found in schema')
    return schema.types[schema.get_field_index(type_name)]


def create_scalar(value, schema, type_name):
    """
    create a scalar from a python or numpy value

    :param schema: schema to use
    :param type_name: name of the type
    :param value: the input value
    :return: arrow scalar
    """
    dtype = name_to_type(schema, type_name)
    return pa.scalar(value, type=dtype)


def create_array(array, schema, type_name):
    """
    create an arrow array from a python or numpy array-like

    :param schema: schema to use
    :param type_name: name of the type
    :param array: the input array
    :return: arrow array
    """
    dtype = name_to_type(schema, type_name)
    return pa.array(array, type=dtype.value_type)


def set_field_int_metadata(schema, field: str, name: str, value: int):
    """
    returns arrow schema with metadata with name set to int value
    in the metadata of field

    :param schema: input schema
    :param field: name of field
    :param name: name of metadata
    :param value: value of field (will be stored as bytes)
    :return: new schema updated from input schema
    """
    for i in schema.get_all_field_indices(field):
        schema = schema.set(i, schema[i].with_metadata({name: value.to_bytes(8, byteorder='big')}))
    return schema


def get_field_int_metadata(field, name: str) -> int:
    """
    returns int encoded in metadata of field

    :param field: arrow field, e.g. table.field('fp')
    :param name: name of metadata
    :return: value of field
    """
    return int.from_bytes(field.metadata[name.encode()], 'big')


def massinfo2struct(mass_info):
    """
    put mass_info data into a pyarrow StructArray

    :param mass_info: mass_info
    :return: StructArray
    """
    return pa.StructArray.from_arrays([[mass_info.tolerance],
                                       [mass_info.tolerance_type],
                                       [mass_info.mass_type],
                                       [mass_info.neutral_loss],
                                       [mass_info.neutral_loss_charge],
                                       [mass_info.evenly_spaced]], fields=massinfo_struct)


# :param tolerance: mass tolerance.  If 0.5 daltons, this is unit mass
# :param tolerance_type: type of tolerance: "ppm", "daltons"
# :param mass_type: "monoisotopic" or "average"
# :param neutral_loss: # neutral loss chemical formula
# :param neutral_loss_charge: sign of neutral loss
# :param evenly_spaced: spectra m/z values are evenly spaced

massinfo_struct_fields = \
    [
        ("tolerance", pa.float64()),
        ("tolerance_type", pa.dictionary(pa.int8(), pa.string())),
        # ("tolerance_type", pa.string()),
        ("mass_type", pa.dictionary(pa.int8(), pa.string())),
        # ("mass_type", pa.string()),
        # ("tolerance_type", pa.string()),
        # ("mass_type", pa.string()),
        ("neutral_loss", pa.string()),
        ("neutral_loss_charge", pa.int64()),
        ("evenly_spaced", pa.bool_())
    ]

massinfo_struct = pa.struct(massinfo_struct_fields)

# ion annotation
# types of annotations supported:
# peptide/polymer ions due to a single bond break, with ion_type being one of a,b,c,x,y,z.  position specifies break.
# peptide/polymer ions with two bond breaks, eg peptide internal ion.  end_position specifies second break.
# parent ion.
# fragment ion.  fragment is specified using canonical SMILES per rdkit.Chem.rdmolfiles.MolFragmentToSmiles
ion_annot_fields = \
    [
        ("ion_type", pa.dictionary(pa.int32(), pa.string())),  # ion type or canonical SMILES
        ("product_charge", pa.int8()),  # charge of ion
        ("isotope", pa.uint8()),   # which isotopic peak?  0 is monoisotopic
        ("ion_subtype", pa.dictionary(pa.int32(), pa.string())),  # subtype of ion_type
        ("position", pa.uint16()),  # position of bond break in polymer
        ("end_position", pa.uint16()),   # optional end position of bond break in polymer (eg internal ion)
        ("aa_before", pa.dictionary(pa.int8(), pa.string())),  # amino acid before cleavage point
        ("aa_after", pa.dictionary(pa.int8(), pa.string())),  # amino acid after cleavage point
        ("ptm_before", pa.dictionary(pa.int16(), pa.string())),  # ptm before cleavage point
        ("ptm_after", pa.dictionary(pa.int16(), pa.string())),  # ptm after cleavage point
    ]

ion_annot = pa.struct(ion_annot_fields)

# Note: prefer to use pa.large_list over pa.list_ for large data as large_list uses 64 bit offsets
# and pa.list_ uses 32 bit offsets, which overflows for large spectra libraries

# experimental metadata fields shared by all types of experiments
base_experimental_fields = [
    pa.field("id", pa.uint64()),
    pa.field("charge", pa.int8()),
    pa.field("ev", pa.float64(), metadata={'description': 'collision energy (voltage drop to collision cell)'}),
    pa.field("instrument", pa.string()),
    pa.field("instrument_type", pa.string()),
    pa.field("instrument_model", pa.string()),
    pa.field("ion_mode", pa.string()),
    pa.field("ionization", pa.string()),
    pa.field("name", pa.string()),
    pa.field("synonyms", pa.string()),
    pa.field("scan", pa.string()),
    pa.field("nce", pa.float64(), metadata={'description': 'normalized collision energy'}),
    pa.field("collision_energy", pa.float64(), metadata={'description': 'collision energy, either ev or calculated from nce'}),
    pa.field("retention_time", pa.float64(), metadata={'description': 'retention time in seconds'}),
    pa.field("collision_gas", pa.string()),
    pa.field("insource_voltage", pa.int64()),
    pa.field("sample_inlet", pa.string()),
]

# spectrum fields shared by all types of experiments
base_spectrum_fields = [
    pa.field("intensity", pa.large_list(pa.float64())),
    pa.field("stddev", pa.large_list(pa.float64())),
    pa.field("product_massinfo", massinfo_struct),
    pa.field("mz", pa.large_list(pa.float64())),
    pa.field("precursor_intensity", pa.float64()),
    pa.field("precursor_massinfo", massinfo_struct),
    pa.field("precursor_mz", pa.float64()),
    pa.field("starts", pa.large_list(pa.float64())),
    pa.field("stops", pa.large_list(pa.float64())),
]

# annotations on spectra, shared by all types of experiments
base_annotation_fields = [
    pa.field("exact_mass", pa.float64()),
    pa.field("exact_mw", pa.float64()),
    pa.field("annotations", pa.large_list(ion_annot)),
    pa.field("spectrum_fp", pa.large_list(pa.uint8())),
    pa.field("spectrum_fp_count", pa.int32()),
    pa.field("hybrid_fp", pa.large_list(pa.float32())),
    pa.field("set", pa.dictionary(pa.int8(), pa.string())),
    pa.field("composition", pa.dictionary(pa.int8(), pa.string())),  # bestof, consensus
]

base_fields = base_experimental_fields + base_spectrum_fields + base_annotation_fields
base_schema = pa.schema(base_fields)

# fields used in peptide experiments that define the experimental molecule
peptide_fields = [
    pa.field("peptide", pa.string()),
    pa.field("peptide_len", pa.int32()),
    # note that mod_names should be a list of dictionary arrays but it's not clear how to initialize
    # this properly with arrays of mod_names, such as in library import.  So for now, just using a list of int16
    # which matches the modifications dictionary in encoding.py
    pa.field("mod_names", pa.large_list(pa.int16())),  # should be pa.large_list(pa.dictionary(pa.int16(), pa.string()))
    pa.field("mod_positions", pa.large_list(pa.int32())),
    pa.field("peptide_type", pa.dictionary(pa.int8(), pa.string())),  # tryptic, semitryptic
]
mod_names_field = peptide_fields[1]

# fields used in small molecule experiments that define the experimental molecule
molecule_definition_fields = [
    pa.field("mol", pa.string()),  # rdkit molecule expressed as MolInterchange JSON
]

# experimental metadata fields used in small molecule experiments
molecule_experimental_fields = [
    pa.field("column", pa.string()),
    pa.field("experimental_ri", pa.float64()),
    pa.field("experimental_ri_data", pa.int32()),
    pa.field("experimental_ri_error", pa.float64()),
    pa.field("stdnp", pa.float64()),
    pa.field("stdnp_data", pa.int32()),
    pa.field("stdnp_error", pa.float64()),
    pa.field("stdpolar", pa.float64()),
    pa.field("stdpolar_data", pa.int32()),
    pa.field("stdpolar_error", pa.float64()),
    pa.field("vial_id", pa.int64())
]

# annotation fields used in small molecule experiments
molecule_annotation_fields = [
    pa.field("aromatic_rings", pa.int64()),
    pa.field("ecfp4", pa.large_list(pa.uint8())),
    pa.field("ecfp4_count", pa.int32()),
    pa.field("estimated_ri", pa.float64()),
    pa.field("estimated_ri_error", pa.float64()),
    pa.field("formula", pa.string()),
    pa.field("has_2d", pa.bool_()),
    pa.field("has_conformer", pa.bool_()),
    pa.field("has_tms", pa.int64()),
    pa.field("hba", pa.int64()),
    pa.field("hbd", pa.int64()),
    pa.field("inchi_key", pa.string()),
    pa.field("inchi_key_orig", pa.string()),
    pa.field("isomeric_smiles", pa.string()),
    pa.field("num_atoms", pa.int64()),
    pa.field("num_undef_double", pa.int64()),
    pa.field("num_undef_stereo", pa.int64()),
    pa.field("rotatable_bonds", pa.int64()),
    pa.field("smiles", pa.string()),
    pa.field("tpsa", pa.float64()),
]

molecule_fields = molecule_definition_fields + molecule_annotation_fields + molecule_experimental_fields

experimental_fields = molecule_experimental_fields + peptide_fields + base_experimental_fields

spectrums_schema = pa.schema(base_fields + peptide_fields)
molecules_schema = pa.schema(base_fields + molecule_fields)

# Useful lists of fields
# minimal set of spectrum fields
min_spectrum_fields = ["id", "name", "precursor_mz", "precursor_massinfo",
                       "precursor_intensity", "mz", "product_massinfo", "intensity"]


peak_join_fields = [
    ("ion1", pa.uint16()),
    ("ion2", pa.uint16()),
    ]

peak_join = pa.struct(peak_join_fields)

spectrum_join_fields = [
    ("spectrum1id", pa.uint16()),
    ("spectrum2id", pa.uint16()),
    ("spectrum1field", pa.string()),
    ("spectrum2field", pa.string()),
    ("peak_join", pa.list_(peak_join))
]

spectrum_join = pa.struct(spectrum_join_fields)

join_types = ["exp2predicted", "exp2theo", "predicted2theo"]

join_fields = [
    ("join_type", pa.dictionary(pa.int8(), pa.string())),
    ("join", spectrum_join),
]

join = pa.struct(join_fields)

join_schema = pa.schema([pa.field("join", pa.list_(join))])

# end result should be an array of
# spectrumid1 (exp), spectrumid2 (pred), spectrumid3 (theo), mz for all three, intensity for all 3 (normalized)
# std dev for predicted, annot for theo, precursor for exp, charge for exp, AA before, AA after, eV, peptide length, ptmid as int before, ptmid as int after
# cosine score of exp ver other two, num peaks of all three, num peaks matched to exp.

# # Common
# charge                     int64
# ev                        object
# instrument                object
# instrument_type           object
# ion_mode                  object
# ionization                object
# name                      object
# nce                       object


# # Common, modified
# precursor_mz              object
# precursor_type            object
# spectrum                  object
# spectrum_type             object


# # molecules only
# aromatic_rings             int64
# collision_energy           int64
# collision_gas             object
# column                    object
# ecfp4                     object
# ecfp4_count                int64
# estimated_ri              object
# estimated_ri_error        object
# exact_mass               float64
# exact_mw                 float64
# experimental_ri           object
# experimental_ri_data      object
# experimental_ri_error     object
# formula                   object
# has_2d                      bool
# has_conformer               bool
# has_tms                    int64
# hba                        int64
# hbd                        int64
# inchi_key                 object
# inchi_key_orig            object
# insource_voltage           int64
# isomeric_smiles           object
# mol                       object
# num_atoms                  int64
# num_undef_double           int64
# num_undef_stereo           int64
# rotatable_bonds            int64
# sample_inlet              object
# set                       object
# smiles                    object
# stdnp                     object
# stdnp_data                object
# stdnp_error               object
# stdpolar                  object
# stdpolar_data             object
# stdpolar_error            object
# synonyms                  object
# tpsa                     float64
# vial_id                    int64

hitlist_fields = [
    pa.field("peptide", pa.string()),
    pa.field("mod_names", pa.large_list(pa.int32())),
    pa.field("mod_positions", pa.large_list(pa.int32())),
    pa.field("charge", pa.int8()),
    #pa.field("ev", pa.float64(), metadata={'description': 'collision energy (voltage drop to collision cell)'}),
    pa.field("query_id", pa.uint64()),
    pa.field("hit_id", pa.uint64()),
    pa.field("accession", pa.string()),
    pa.field("protein_start", pa.uint16()),
    pa.field("protein_stop", pa.uint16()),
    pa.field("source_search", pa.string()),
    pa.field("decoy_hit", pa.bool_()),
    pa.field("raw_score", pa.float32())
    # pa.field("Andromeda PEP", pa.float32())
]

hitlist_schema = pa.schema(hitlist_fields)
