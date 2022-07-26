from massspec.utils.files import (
    spectra_to_msp,
    spectra_to_mgf,
)
import pandas as pd
from massspec.constants import EPSILON
try:
    from IPython import display
except ImportError:
    display = None

@pd.api.extensions.register_dataframe_accessor("lib")
class LibraryAccessor:
    """
    the base pandas accessor class.  To use the accessor, import LibraryAccessor into your code.

    Notes:
    = to add info to the dataframe itself, use  _obj.tandem_peptide_library.__dict__['info'] = info
    - in the future, by caching the record id, the serialization functions can be modified to read in chunks
    """

    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj  # the dataframe
        if 'lib_type' not in self._obj.attrs:
            self._obj.attrs['lib_type'] = None  # used to persist the type of library. Unfortunately does not persist
        # across all operations, e.g. query()

    def copy(self):
        return self._obj

    @staticmethod
    def _validate(pandas_obj):
        # verify this is a DataFrame
        if not isinstance(pandas_obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")

    def to_msp(self, fp, spectrum_column='spectrum', annotate=False, ion_types=None):
        """
        write to an msp file

        :param fp: stream or filename
        :param spectrum_column: name of the column containing the spectrum
        :param annotate: annotate the spectrum
        :param ion_types: ion types to annotate
        """
        spectra_to_msp(fp, self._obj[spectrum_column], annotate=annotate, ion_types=ion_types)
        
    def to_mgf(self, fp, spectrum_column='spectrum'):
        """
        write to an mgf file

        :param fp: stream or filename
        :param spectrum_column: name of the column containing the spectrum
        """
        spectra_to_mgf(fp, self._obj[spectrum_column])
   
    def display(self):
        if display is not None:
            return display.display(display.HTML(self._obj.to_html(escape=False, index=False, float_format="%.2f")))
        else:
            return self._obj
