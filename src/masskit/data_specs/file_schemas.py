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

# nested schemas
nested_peptide_spectrum_schema = pa.schema(base_property_fields + peptide_property_fields + peptide_spectrum_field)
nested_molecule_spectrum_schema = pa.schema(base_property_fields + molecule_property_fields + molecule_definition_fields + molecule_spectrum_field)

# fields useful for dropping from csv output
csv_drop_fields = ['mol', 'shortest_paths', 'spectrum', 'predicted_spectrum', 'theoretical_spectrum', 'spectrum_fp', 'spectrum_fp_count', 'ecfp4', 'ecfp4_count']
