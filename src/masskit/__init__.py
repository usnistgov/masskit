import pyarrow
from .data_specs.arrow_types import MolArrowType, JSONArrowType, SpectrumArrowType
from .data_specs.schemas import molecules_struct

pyarrow.register_extension_type(MolArrowType())
pyarrow.register_extension_type(JSONArrowType())
pyarrow.register_extension_type(SpectrumArrowType(molecules_struct))
# apparently only need to register once, irrespective of parameterized type
# pa.register_extension_type(SpectrumArrowType(mkschemas.peptide_struct))
