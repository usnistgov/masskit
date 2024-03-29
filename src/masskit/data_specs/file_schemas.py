import pyarrow as pa
# from masskit.data_specs.schemas import *

from ..data_specs import arrow_types as mkarrow_types
from ..data_specs import schemas as mkschemas

"""
Separate out fields that depend on python objects to avoid circular references
"""

# fields used in small molecule experiments that define the experimental molecule
molecule_definition_fields = [
    pa.field("mol", mkarrow_types.MolArrowType()),  # rdkit molecule expressed as MolInterchange JSON
    pa.field("shortest_paths", mkarrow_types.PathArrowType()),  # shortest paths between atoms in molecule
]

peptide_spectrum_field = [pa.field("spectrum", mkarrow_types.SpectrumArrowType(storage_type=mkschemas.molecules_struct))]
molecule_spectrum_field = [pa.field("spectrum", mkarrow_types.SpectrumArrowType(storage_type=mkschemas.peptide_struct))]

# flattened schema for standard spectral file formats which do not include a molecular connectivity graph
flat_peptide_schema = pa.schema(mkschemas.compose_fields(mkschemas.base_fields, mkschemas.peptide_property_fields))

# flattened schema for files that include spectral data and molecular connectivity graphs, e.g. sdf/mol files.
flat_molecule_schema = pa.schema(mkschemas.compose_fields(mkschemas.base_fields, mkschemas.molecule_property_fields, molecule_definition_fields))

# nested schemas
nested_peptide_spectrum_schema = pa.schema(mkschemas.base_property_fields + mkschemas.peptide_property_fields + peptide_spectrum_field)
nested_molecule_spectrum_schema = pa.schema(mkschemas.base_property_fields + mkschemas.molecule_property_fields + molecule_definition_fields + molecule_spectrum_field)

# fields useful for dropping from csv output
csv_drop_fields = ['mol', 'shortest_paths', 'spectrum', 'predicted_spectrum', 'theoretical_spectrum', 'spectrum_fp', 'spectrum_fp_count', 'ecfp4', 'ecfp4_count']
# fields useful for dropping out for display
display_drop_fields = ['mol', 'shortest_paths', 'spectrum_fp', 'spectrum_fp_count', 'ecfp4', 'ecfp4_count']

# groups of related schemas
schema_groups = {
    "peptide": {"storage_type": mkschemas.peptide_struct,
                "flat_schema": flat_peptide_schema,
                "nested_schema": nested_peptide_spectrum_schema},
    "mol": {"storage_type": mkschemas.molecules_struct,
            "flat_schema": flat_molecule_schema,
            "nested_schema": nested_molecule_spectrum_schema},
}
