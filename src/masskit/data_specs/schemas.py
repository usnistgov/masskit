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


def compose_fields(*field_lists):
    """
    Compose field lists, retaining order but removing redundancies

    :param field_lists: variable number of field lists
    :return: combined noin-redundant field list
    """
    unique_fields = {}
    ret_fields = []
    for field_list in field_lists:
        for field in field_list:
            if field.name not in unique_fields:
                ret_fields.append(field)
                unique_fields[field.name] = field
            else:
                if field != unique_fields[field.name]:
                    # duplicate name but different type
                    raise TypeError
    return ret_fields


def subtract_fields(field_list, fields2bsubtracted):
    """
    delete fields from a field list

    :param field_list: field list to be edited
    :param fields2bsubtracted: list of fields to be deleted
    :return: edited field list
    """
    ret_fields = []
    for field in field_list:
        if field not in fields2bsubtracted:
            ret_fields.append(field)
    return ret_fields


# :param tolerance: mass tolerance.  If 0.5 daltons, this is unit mass
# :param tolerance_type: type of tolerance: "ppm", "daltons"
# :param mass_type: "monoisotopic" or "average"
# :param neutral_loss: # neutral loss chemical formula
# :param neutral_loss_charge: sign of neutral loss
# :param evenly_spaced: spectra m/z values are evenly spaced

massinfo_struct_fields = \
    [
        ("tolerance", pa.float64()),
        ("tolerance_type", pa.dictionary(pa.int32(), pa.string())),
        # ("tolerance_type", pa.string()),
        ("mass_type", pa.dictionary(pa.int32(), pa.string())),
        # ("mass_type", pa.string()),
        # ("tolerance_type", pa.string()),
        # ("mass_type", pa.string()),
        ("neutral_loss", pa.string()),
        ("neutral_loss_charge", pa.int16()),
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
        ("product_charge", pa.int16()),  # charge of ion
        ("isotope", pa.uint8()),   # which isotopic peak?  0 is monoisotopic
        ("ion_subtype", pa.dictionary(pa.int32(), pa.string())),  # subtype of ion_type
        ("position", pa.uint16()),  # position of bond break in polymer
        ("end_position", pa.uint16()),   # optional end position of bond break in polymer (eg internal ion)
        ("aa_before", pa.dictionary(pa.int32(), pa.string())),  # amino acid before cleavage point
        ("aa_after", pa.dictionary(pa.int32(), pa.string())),  # amino acid after cleavage point
        ("ptm_before", pa.dictionary(pa.int32(), pa.string())),  # ptm before cleavage point
        ("ptm_after", pa.dictionary(pa.int32(), pa.string())),  # ptm after cleavage point
    ]

ion_annot = pa.struct(ion_annot_fields)

# Note: prefer to use pa.large_list over pa.list_ for large data as large_list uses 64 bit offsets
# and pa.list_ uses 32 bit offsets, which overflows for large spectra libraries

# experimental metadata fields shared by all types of experiments
min_fields = [
    pa.field("id", pa.uint64()),
]

# experimental metadata fields shared by many types of experiments
base_experimental_fields = [
    pa.field("instrument", pa.string()),
    pa.field("instrument_type", pa.string()),
    pa.field("instrument_model", pa.string()),
    pa.field("ion_mode", pa.string()),
    pa.field("ionization", pa.string()),
    pa.field("name", pa.string()),
    pa.field("casno", pa.string()),
    pa.field("synonyms", pa.string()),
    pa.field("scan", pa.string()),
    pa.field("collision_energy", pa.float32(), metadata={'description': 'collision energy, either ev or calculated from nce'}),
    pa.field("retention_time", pa.float64(), metadata={'description': 'retention time in seconds'}),
    pa.field("collision_gas", pa.string()),
    pa.field("insource_voltage", pa.int64()),
    pa.field("sample_inlet", pa.string()),
    pa.field("ev", pa.float32(), metadata={'description': 'collision energy (voltage drop to collision cell)'}),
    pa.field("nce", pa.float32(), metadata={'description': 'normalized collision energy'})
]

# large measured spectrum fields shared by all types of experiments
base_spectrum_large_fields = [
    pa.field("intensity", pa.large_list(pa.float32())),
    pa.field("stddev", pa.large_list(pa.float64())),
    pa.field("product_massinfo", massinfo_struct),
    pa.field("mz", pa.large_list(pa.float64())),
    pa.field("precursor_intensity", pa.float64()),
    pa.field("precursor_massinfo", massinfo_struct),
    pa.field("tolerance", pa.large_list(pa.float64())),  # mz tolerance
]

# small measured spectrum fields shared by all types of experiments
precursor_mz_field = pa.field("precursor_mz", pa.float64())
base_spectrum_small_fields = [
    pa.field("charge", pa.int16()),
    precursor_mz_field,
]

# measured spectrum fields shared by all types of experiments
base_spectrum_fields = base_spectrum_large_fields + base_spectrum_small_fields

# annotations on spectra, shared by all types of experiments

base_annotation_small_fields = [
    pa.field("exact_mass", pa.float64()),
    pa.field("exact_mw", pa.float64()),
    pa.field("set", pa.dictionary(pa.int32(), pa.string())),
    pa.field("composition", pa.dictionary(pa.int32(), pa.string())),  # bestof, consensus
    pa.field("spectrum_fp", pa.large_list(pa.uint8()), metadata={b"fp_size": (2000).to_bytes(8, byteorder='big')}),
    pa.field("spectrum_fp_count", pa.int32()),
]

base_annotation_large_fields = [
    pa.field("annotations", pa.large_list(ion_annot)),
]

base_annotation_fields = base_annotation_small_fields + base_annotation_large_fields

# base experimental metadata
base_metadata_fields = base_experimental_fields

# fields treated as properties (no large fields)
base_property_fields = compose_fields(min_fields, base_metadata_fields, base_spectrum_small_fields, base_annotation_small_fields)

# all basic spectrum fields, flattened
base_fields = compose_fields(base_property_fields, base_annotation_large_fields, base_spectrum_large_fields)
base_schema = pa.schema(base_fields)

# fields used in peptide experiments that define the experimental molecule
mod_names_field = pa.field("mod_names", pa.large_list(pa.int16()))  # should be pa.large_list(pa.dictionary(pa.int16(), pa.string()))
peptide_definition_fields = [
    pa.field("peptide", pa.string()),
    pa.field("peptide_len", pa.int32()),
    pa.field("peptide_type", pa.dictionary(pa.int32(), pa.string())),  # tryptic, semitryptic
    # note that mod_names should be a list of dictionary arrays but it's not clear how to initialize
    # this properly with arrays of mod_names, such as in library import.  So for now, just using a list of int16
    # which matches the modifications dictionary in encoding.py
    mod_names_field,
    pa.field("mod_positions", pa.large_list(pa.int32())),
]

# peptide experimental metadata
protein_id_field = [pa.field("protein_id", pa.large_list(pa.string()))]
peptide_metadata_fields = compose_fields(peptide_definition_fields, protein_id_field)

# all property fields that are specific for describing peptides
peptide_property_fields = peptide_metadata_fields

# experimental metadata fields used in small molecule experiments, measured or recorded
molecule_experimental_fields = [
    pa.field("column", pa.string()),
    pa.field("experimental_ri", pa.float32()),
    pa.field("experimental_ri_data", pa.int32()),
    pa.field("experimental_ri_error", pa.float32()),
    pa.field("stdnp", pa.float32()),
    pa.field("stdnp_data", pa.int32()),
    pa.field("stdnp_error", pa.float32()),
    pa.field("stdpolar", pa.float32()),
    pa.field("stdpolar_data", pa.int32()),
    pa.field("stdpolar_error", pa.float32()),
    pa.field("vial_id", pa.int64())
]

# annotation fields used in small molecule experiments that are calculated from structure
molecule_annotation_fields = [
    pa.field("aromatic_rings", pa.int32()),
    pa.field("ecfp4", pa.large_list(pa.uint8()), metadata={b"fp_size": (4096).to_bytes(8, byteorder='big')}),
    pa.field("ecfp4_count", pa.int32()),
    pa.field("estimated_ri", pa.float32()),  # standard semipolar
    pa.field("estimated_ri_error", pa.float32()),  
    pa.field("estimated_ri_stdnp", pa.float32()),  # standard nonpolar
    pa.field("estimated_ri_stdnp_error", pa.float32()),
    pa.field("estimated_ri_stdpolar", pa.float32()),  # standard polar
    pa.field("estimated_ri_stdpolar_error", pa.float32()),
    pa.field("formula", pa.string()),
    pa.field("has_2d", pa.bool_()),
    pa.field("has_conformer", pa.bool_()),
    pa.field("has_tms", pa.int32()),
    pa.field("hba", pa.int32()),
    pa.field("hbd", pa.int32()),
    pa.field("inchi_key", pa.string()),
    pa.field("inchi_key_orig", pa.string()),
    pa.field("isomeric_smiles", pa.string()),
    pa.field("num_atoms", pa.int32()),
    pa.field("num_undef_double", pa.int32()),
    pa.field("num_undef_stereo", pa.int32()),
    pa.field("rotatable_bonds", pa.int32()),
    pa.field("smiles", pa.string()),
    pa.field("tpsa", pa.float32()),
    pa.field("logp", pa.float32()),
    pa.field("fragments", pa.int32()),  # number of unbonded molecules in Mol
]

# small molecule experimental metadata
molecule_metadata_fields = molecule_experimental_fields

# all property fields that describe small molecules and associated spectra.  used in file schemas
molecule_property_fields = compose_fields(molecule_metadata_fields, molecule_annotation_fields)

# all property fields. Use to populate properties in spectra, Accumulators, and columns in arrow tables.
property_fields = compose_fields(base_property_fields, peptide_property_fields, molecule_property_fields)
property_schema = pa.schema(property_fields)

accumulator_fields = [
    pa.field('predicted_mean', pa.float64()),
    pa.field('predicted_stddev', pa.float64()),
    ]
# all property fields plus accumulator fields.
accumulator_property_fields = compose_fields(property_fields, accumulator_fields)

spectrum_accumulator_fields = [
    pa.field('cosine_score', pa.float32()),
]

# struct for peptide spectra
peptide_struct = pa.struct(compose_fields(base_fields, peptide_property_fields))

# struct for small molecule spectrum
# struct doesn't contain the mol or path data as those are separate columns
molecules_struct = pa.struct(compose_fields(base_fields, molecule_property_fields))

# generic spectrum expressed as a struct
spectrum_struct = pa.struct(compose_fields(base_fields, property_fields))

# Useful lists of fields
# minimal set of spectrum fields
min_spectrum_field_names = ["id", "name", "precursor_mz", "precursor_massinfo",
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
    ("join_type", pa.dictionary(pa.int32(), pa.string())),
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
    pa.field("charge", pa.int16()),
    #pa.field("ev", pa.float64(), metadata={'description': 'collision energy (voltage drop to collision cell)'}),
    pa.field("query_id", pa.uint64()),
    pa.field("hit_id", pa.uint64()),
    pa.field("accession", pa.string()),
    pa.field("protein_start", pa.uint32()),
    pa.field("protein_stop", pa.uint32()),
    pa.field("source_search", pa.string()),
    pa.field("decoy_hit", pa.bool_()),
    pa.field("raw_score", pa.float32())
    # pa.field("Andromeda PEP", pa.float32())
]

hitlist_schema = pa.schema(hitlist_fields)


def create_getter(name):
    """ 
    create a generic property getter on the props dictionary of an object

    :param name: name of the property
    :return: getter function
    """
    def getter(self):
        return self.props.get(name, None)
    return getter


def create_setter(name):
    """ 
    create a generic property setter on the props dictionary of an object

    :param name: name of the property
    :return: setter function
    """
    def setter(self, data):
        self.props[name] = data
    return setter


def populate_properties(class_in, fields=property_fields):
    """
    given a class (or any object), create a set of properties from a list of fields
    
    :param class_in: the class/object to be modified
    :param fields: a list of pyarrow fields whose names will be used to create properties
    """
    for field in fields:
        setattr(class_in, field.name, property(create_getter(field.name),create_setter(field.name)))
