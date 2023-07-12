import pandas as pd
try:
    from IPython import display
except ImportError:
    display = None

def display_masskit_df(df):
    if display is not None:
        return display.display(display.HTML(df.to_html(escape=False, index=False, float_format="%.2f")))
    else:
        return df

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
        # across all operations, e.g. query()

    def copy(self):
        return self._obj

    @staticmethod
    def _validate(pandas_obj):
        # verify this is a DataFrame
        if not isinstance(pandas_obj, pd.DataFrame):
            raise AttributeError("Must be a pandas DataFrame")
   
    def display(self):
        if display is not None:
            return display.display(display.HTML(self._obj.to_html(escape=False, index=False, float_format="%.2f")))
        else:
            return self._obj
