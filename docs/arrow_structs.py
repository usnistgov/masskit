#!python
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time
import string
import sys
from types import MethodType

ALPHABET = np.array(list(string.ascii_letters))


def random_string(length):
    return "".join(np.random.choice(ALPHABET, size=length))


def create_dataset(rows=1500, int_cols=5, float_cols=5, str_cols=2, list_cols=1):
    rng = np.random.default_rng()

    b = 0
    e = int_cols
    names = list(string.ascii_letters[b:e])
    ic = pd.DataFrame(rng.integers(0, 100, size=(rows, int_cols)), columns=names)

    b = e
    e = e + float_cols
    names = list(string.ascii_letters[b:e])
    fc = pd.DataFrame(
        rng.random(rows * float_cols).reshape(rows, float_cols), columns=names
    )

    b = e
    e = e + str_cols
    names = list(string.ascii_letters[b:e])
    vfunc = np.vectorize(random_string)
    rand_lengths = rng.integers(5, 25, size=(rows, str_cols))
    sc = pd.DataFrame(vfunc(rand_lengths), columns=names)

    table = pa.table(pd.concat([ic, fc, sc], axis=1))

    b = e
    e = e + list_cols
    names = list(string.ascii_letters[b:e])
    lc = []
    for name in names:
        larr = []
        for j in range(rows):
            rand_length = rng.integers(5, 25, size=1)
            lst = rng.random(rand_length)
            larr.append(lst)
        table = table.append_column(name, pa.array(larr))

    # sys.exit()
    return table


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


def row_view(table, idx=0):
    class row_accessor:
        def __init__(self, table, idx):
            self.table = table
            self.idx = idx

    def add_accessor_fn(inst, name, type):
        fn_name = "get_" + name

        def fn_list(self):
            return self.table.column(name).slice(self.idx, 1)[0].values.to_numpy()

        def fn_scalar(self):
            return self.table.column(name).slice(self.idx, 1)[0].as_py()

        def fn(self):
            return self.table.column(name).slice(self.idx, 1)[0]

        if pa.types.is_list(type):
            setattr(inst, fn_name, MethodType(fn_list, inst))
        elif pa.types.is_boolean(type):
            setattr(inst, fn_name, MethodType(fn_scalar, inst))
        elif pa.types.is_integer(type):
            setattr(inst, fn_name, MethodType(fn_scalar, inst))
        elif pa.types.is_floating(type):
            setattr(inst, fn_name, MethodType(fn_scalar, inst))
        elif pa.types.is_string(type):
            setattr(inst, fn_name, MethodType(fn_scalar, inst))
        else:
            setattr(inst, fn_name, MethodType(fn, inst))

    inst = row_accessor(table, idx)
    for col in table.column_names:
        add_accessor_fn(inst, col, table.column(col).type)

    return inst


def display(df):
    print(df.shape)
    if df.shape[0] <= 10:
        print(df)
    else:
        print(df.head(5))
        print("---")
        print(df.tail(5).to_string(header=False))


if __name__ == "__main__":

    # df = create_dataframe(rows=13)
    start = time.time()
    # table = create_dataset(rows=500, int_cols=5, float_cols=4, str_cols=2)
    table = create_dataset(rows=5, int_cols=1, float_cols=1, str_cols=1)
    table = create_dataset(rows=5, int_cols=1, float_cols=1, str_cols=1, list_cols=2)
    end = time.time()
    # display(df)
    # print(table.schema)
    print(f"Shape: {table.shape}")
    print(f"MB: {table.nbytes/(1024*1024):.3f}")
    print(f"Create dataset time elapsed: {end - start:.3f}")
    print(table.columns)
    print(table.column_names)
    print(table.schema)
    # print(st)

    sys.exit()

    # Timing functions

    start = time.time()
    pq.write_table(table, "junk_test.parquet", version="2.0")
    end = time.time()
    print(f"Write dataset time elapsed: {end - start:.3f}")

    start = time.time()
    table2 = pq.read_table("junk_test.parquet")
    end = time.time()
    print(f"Read dataset time elapsed: {end - start:.3f}")

    start = time.time()
    st = table_to_struct(table2)
    end = time.time()
    print(f"Dataset to struct time elapsed: {end - start:.3f}")

    print(st.to_pandas())
    # display(st)

    start = time.time()
    table3 = struct_to_table(st)
    end = time.time()
    print(f"Struct to dataset time elapsed: {end - start:.3f}")

    # print(table3.to_pandas())
    display(table3)
