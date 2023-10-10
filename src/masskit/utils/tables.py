import itertools
import string
from types import MethodType

import numpy as np
import pyarrow as pa

from ..data_specs import arrow_types as _mkarrow_types

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


# def table_to_struct(table: pa.Table) -> pa.StructArray:
#     fields, arrs = [], []
#     for column_index in range(table.num_columns):
#         fields.append(table.field(column_index))
#         arrs.append(table.column(column_index).flatten()[0].chunks[0])
#     return pa.StructArray.from_arrays(arrs, fields=fields)

def table_to_structarray(table: pa.Table, structarray_type:pa.ExtensionType=None) -> pa.StructArray:
    """
    convert a spectrum table into a struct array.
    if an ExtensionType is passed in, will create a struct array of that type

    :param table: spectrum table
    :param structarray_type: the type of the array returned, e.g. SpectrumArrowType()
    :return: StructArray
    """
    
    # combine chunks as StructArray.from_arrays() only takes Arrays, not ChunkedArray
    table = table.combine_chunks()
    # TODO: this should be modified to work on ChunkedArrays to created ChunkedArray of Struct
    # to improve performance, or Structs should be created in the first place
    arrays = [table.column(i).chunk(0) for i in range(table.num_columns)]
    column_names = table.column_names

    output = pa.StructArray.from_arrays(arrays, names=column_names)
    if structarray_type is not None:
        output = pa.ExtensionArray.from_storage(structarray_type(storage_type=output.type), output)
    return output


def structarray_to_table(struct: pa.StructArray) -> pa.Table:
    names, arrs = [], []
    for x in struct[0].keys():
        names.append(x)
        arrs.append(struct.field(x))
    return pa.table(arrs, names)


def optimize_structarray(struct: pa.ChunkedArray) -> pa.ChunkedArray:
    flat_struct = struct.combine_chunks()
    if hasattr(flat_struct, 'storage'):
        flat_struct = flat_struct.storage
    table = structarray_to_table(flat_struct)
    table = optimize_table(table)
    # TODO: Need to find a better way to identify an extension array
    extension_type = None
    if "SpectrumArrowArray" in str(struct.chunk(0).__class__):
        extension_type = _mkarrow_types.SpectrumArrowType
    flat_struct = table_to_structarray(table, structarray_type=extension_type)
    return flat_struct

def is_struct_or_extstruct(chunked_array: pa.ChunkedArray) -> bool:
    if pa.types.is_struct(chunked_array.chunk(0).type):
        return True
    if hasattr(chunked_array.chunk(0), 'storage'):
        if pa.types.is_struct(chunked_array.chunk(0).storage.type):
            return True
    return False

def optimize_table(table: pa.Table) -> pa.Table:
    drops = []
    for idx, name, col in zip(itertools.count(), table.column_names, table.columns):
        if col.length() == col.null_count:
            drops.append(name)
        elif is_struct_or_extstruct(col):
            new_struct = optimize_structarray(col)
            table = table.set_column(idx, name, new_struct)
    return table.drop_columns(drops)


def table_add_structarray(table: pa.Table, structarray: pa.StructArray, column_name:str=None) -> pa.Table:  
    """
    add a struct array to a table

    :param table: table to be added to
    :param structarray: structarray to add to table
    :param column_name: name of column to add
    :return: new table with appended column
    """
    if column_name is None:
        column_name = 'spectrum'
    table = table.append_column(column_name, structarray)
    return table

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
                try:
                    return values.to_numpy()
                except pa.ArrowInvalid:
                    return values.to_numpy(zero_copy_only=False)
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


if __name__ == "__main__":

    table = create_dataset(cols=[int, float, list])
    st = table_to_structarray(create_dataset(names=["john", "paul", "george", "ringo"]))
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
