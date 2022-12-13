import numpy as np
import pyarrow as pa
import pandas as pd
from masskit.spectrum.spectrum import init_spectrum
import masskit.data_specs.schemas as ms_schemas
import string
import string
from types import MethodType

ALPHABET = np.array(list(string.ascii_letters))


def random_string(length):
    return "".join(np.random.choice(ALPHABET, size=length))


def create_dataset(rows=5, cols=[int, float, str, list], names=ALPHABET):
    rng = np.random.default_rng()

    data = {}
    for i in range(len(cols)):
        if cols[i] == int:
            data[names[i]] = rng.integers(0, 100, size=rows)
        elif cols[i] == float:
            data[names[i]] = rng.random(size=rows)
        elif cols[i] == str:
            vfunc = np.vectorize(random_string)
            rand_lengths = rng.integers(5, 25, size=rows)
            data[names[i]] = vfunc(rand_lengths)
        elif cols[i] == list:
            larr = []
            for j in range(rows):
                rand_length = rng.integers(5, 25, size=1)
                lst = rng.random(rand_length)
                larr.append(lst)
            data[names[i]] = larr
    return pa.table(data)


def table_to_struct(table):
    fields, arrs = [], []
    for column_index in range(table.num_columns):
        fields.append(table.field(column_index))
        arrs.append(table.column(column_index).flatten()[0].chunks[0])
    return pa.StructArray.from_arrays(arrs, fields=fields)


def struct_to_table(struct):
    fields, arrs = [], []
    for x in struct.slice(0, 1)[0].keys():
        fields.append(x)
        arrs.append(st.field(x))
    return pa.table(arrs, fields)


def struct_view(struct_name, parent):
    class struct_accessor:
        def __init__(self, name, parent):
            self.struct_name = name
            self.parent = parent

        def struct(self):
            return self.parent.table.column(self.struct_name).slice(self.parent.idx, 1)[0]

        def get(self, str):
            if hasattr(self, str):
                attr = getattr(self, str)
                if attr == MethodType:
                    return attr()
                return attr
            return None

    def add_accessor_fn(inst, name, type):
        fn_name = name

        def fn_list(self):
            values = self.struct().get(name).values
            if values:
                return values.to_numpy()
            return None

        def fn_scalar(self):
            return self.struct().get(name).as_py()

        if pa.types.is_list(type) or pa.types.is_large_list(type):
            setattr(inst, fn_name, MethodType(fn_list, inst))
        elif pa.types.is_struct(type):
            raise NotImplementedError
        else:
            setattr(inst, fn_name, MethodType(fn_scalar, inst))

    inst = struct_accessor(struct_name, parent)
    for name, val in inst.struct().items():
        add_accessor_fn(inst, name, val.type)

    return inst


def row_view(table, idx=0):
    class row_accessor:
        def __init__(self, table, idx):
            self.table = table
            self.idx = idx

        def get(self, str):
            if hasattr(self, str):
                attr = getattr(self, str)
                if attr == MethodType:
                    return attr()
                return attr
            return None

    def add_accessor_fn(inst, name, type):
        fn_name = name

        def fn_list(self):
            values = self.table.column(name).slice(self.idx, 1)[0].values
            if values:
                return values.to_numpy()
            return None

        def fn_scalar(self):
            return self.table.column(name).slice(self.idx, 1)[0].as_py()

        if pa.types.is_list(type) or pa.types.is_large_list(type):
            setattr(inst, fn_name, MethodType(fn_list, inst))
        elif pa.types.is_struct(type):
            sv = struct_view(name, inst)
            setattr(inst, fn_name, sv)
        else:
            setattr(inst, fn_name, MethodType(fn_scalar, inst))

    inst = row_accessor(table, idx)
    for col in table.column_names:
        add_accessor_fn(inst, col, table.column(col).type)

    return inst


def row_view_raw(table, idx=0):
    class row_accessor:
        def __init__(self, table, idx):
            self.table = table
            self.idx = idx

    def add_accessor_fn(inst, name, type):
        fn_name = "get_" + name

        def fn(self):
            return self.table.column(name).slice(self.idx, 1)[0]

        setattr(inst, fn_name, MethodType(fn, inst))

    inst = row_accessor(table, idx)
    for col in table.column_names:
        add_accessor_fn(inst, col, table.column(col).type)

    return inst

def arrow_to_pandas(table):
    """
    converts a pyarrow table to a pandas dataframe with spectrum objects
    :param table: The pyarrow table to be converted
    """
    # Create dataframe of spectrums
    rv = row_view(table)
    spectrums = []
    for i in range(table.num_rows):
        rv.idx = i
        spectrums.append(init_spectrum().from_arrow(rv))
    spectrum_df = pd.DataFrame({
        "id": table['id'].to_pandas(),
        "spectrum": spectrums})

    # dataframe of the remaining subset of columns
    desired_names = list(map(lambda x: x.name,
                             ms_schemas.molecule_experimental_fields +
                             ms_schemas.peptide_fields +
                             ms_schemas.base_experimental_fields))
    common_names = set(table.schema.names) & set(desired_names)
    table_df = table.select(common_names).to_pandas()

    # merge and return the two dataframes
    return pd.merge(spectrum_df, table_df, how='inner', on='id')
    
if __name__ == "__main__":

    table = create_dataset(cols=[int, float, list])
    st = table_to_struct(create_dataset(names=["john", "paul", "george", "ringo"]))
    table = table.append_column("singers", st)
    print(table.to_pandas())

    row = 3
    rv = row_view(table, row)

    print(f"\nRow Id {row}:")
    print(f"   Col a: {rv.a()}")
    print(f"   Col b: {rv.b()}")
    print(f"   Col c: {rv.c()}")
    print(f"   Col singers.john: {rv.singers().john()}")
    print(f"   Col singers.paul: {rv.singers().paul()}")
    print(f"   Col singers.george: {rv.singers().george()}")
    print(f"   Col singers.ringo: {rv.singers().ringo()}")
