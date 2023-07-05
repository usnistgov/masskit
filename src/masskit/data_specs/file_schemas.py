import pyarrow as pa
from masskit.data_specs.arrow_types import MolArrowType, PathArrowType, SpectrumArrowType
from masskit.data_specs.schemas import *

"""
Separate out fields that depend on python objects to avoid circular references
"""

# fields used in small molecule experiments that define the experimental molecule
molecule_definition_fields = [
    pa.field("mol", MolArrowType()),  # rdkit molecule expressed as MolInterchange JSON
    pa.field("shortest_paths", PathArrowType()),  # shortest paths between atoms in molecule
]

peptide_spectrum_field = [pa.field("spectrum", SpectrumArrowType(storage_type=molecules_struct))]
molecule_spectrum_field = [pa.field("spectrum", SpectrumArrowType(storage_type=peptide_struct))]

# flattened schema for standard spectral file formats which do not include a molecular connectivity graph
flat_peptide_schema = pa.schema(compose_fields(base_fields, peptide_property_fields))

# flattened schema for files that include spectral data and molecular connectivity graphs, e.g. sdf/mol files.
flat_molecule_schema = pa.schema(compose_fields(base_fields, molecule_property_fields, molecule_definition_fields))

# fields used for querying on spectral libraries, useful for 
# promoting useful peptide_struct and molecules_struct fields to the top level table
# spectrum_query_fields = molecule_definition_fields + \
#                         property_fields + \
#                         base_annotation_query_fields + \
#                         base_spectrum_small_fields + \
#                         base_experimental_fields + \
#                         min_fields
nested_peptide_spectrum_schema = pa.schema(base_property_fields + peptide_property_fields + peptide_spectrum_field)
nested_molecule_spectrum_schema = pa.schema(base_property_fields + molecule_property_fields + molecule_definition_fields + molecule_spectrum_field)
