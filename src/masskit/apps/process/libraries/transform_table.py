#!/usr/bin/env python
import sys
import os
import tempfile
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.feather as feather

from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeRemainingColumn,
    track,
    TransferSpeedColumn,
)

global_console = Console()
progress = Progress(
    #TextColumn("[bold blue]{task.fields[filename,action]}", justify="right"),
    TextColumn("[bold blue]{task.fields[filename]}: [blue]{task.description}", justify="right"),
    BarColumn(bar_width=None),
    "[progress.percentage]{task.percentage:>3.1f}%",
    "•",
    MofNCompleteColumn(),
    "•",
    TimeRemainingColumn(),
    transient=False,
    console=global_console,
)

_numeric_types = [
    pa.bool_(),
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
    pa.float16(),
    pa.float32(),
    pa.float64()
]

_list_types = [
    pa.list_,
    pa.large_list
]

def str2pyarrow_type(s: str) -> pa.DataType:
    for nt in _numeric_types:
        if s == str(nt):
            return nt

    for lt in _list_types:
        for nt in _numeric_types:
            if s == str(lt(nt)):
                return lt(nt)
    return pa.null()

def cast_columns(table: pa.Table, cfg: DictConfig):
    columns = table.column_names
    for cast in cfg:
        idx = columns.index(cast.field)
        old_column = table[cast.field]
        cast_type = str2pyarrow_type(cast.type)
        table = table.set_column(
            idx,
            cast.field,
            pc.cast(old_column, cast_type)
        )
    return table

def compress_start_stop(table: pa.Table):
    offsets = table["stops"].combine_chunks().offsets
    flat_values = pc.cast(
            pc.divide(
                pc.subtract(
                    pc.list_flatten(
                        table["stops"]
                    ), 
                    pc.list_flatten(
                        table["starts"]
                    )
                ),2
            ).combine_chunks(),
        pa.float32()
    )
    new_values = pa.LargeListArray.from_arrays(offsets, flat_values)
    table = table.drop_columns(['starts', 'stops'])
    table = table.append_column('mz_window', new_values)
    return table


def misc_operations(table: pa.Table, cfg: DictConfig):
    for operation in cfg:
        if operation == "compress_start_stop":
            table = compress_start_stop(table)
    return table

def get_sort_index(table: pa.Table, cfg: DictConfig):
    sort_keys = []
    for skey in cfg:
        sort_keys.append((skey.field, skey.order))
    sortidx = pc.sort_indices(table, sort_keys)
    return sortidx

def process_and_cache(infile: Path, outfile: Path, sort: DictConfig, batch_size=500, casts=None, operations=None) -> Path:
    task_id = progress.add_task(
        "mapping",
        filename=infile.name,
        start=False,
    )
    with pa.memory_map(str(infile), 'rb') as source:
        table = pa.ipc.open_file(source).read_all()

    if (casts):
        progress.update(task_id, description="casting")
        table = cast_columns(table, casts)
    if (operations):
        progress.update(task_id, description="misc ops")
        table = misc_operations(table, operations)

    progress.update(task_id, description="sorting")
    sortidx = get_sort_index(table, sort)
    progress.update(task_id, total=len(sortidx))
    progress.update(task_id, description="caching")
    progress.start_task(task_id)
    with pa.OSFile(str(outfile), 'wb') as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for start in range(0,len(sortidx),batch_size):
                subset = sortidx.slice(start,batch_size)
                batch = table.take(subset)
                writer.write(batch)
                progress.update(task_id, advance=len(batch))
                #print(f"Wrote batch {start}:{start+len(batch)}")

def concat_to_output(files: list, sort: DictConfig, output: DictConfig):
    task_id = progress.add_task(
        "mapping",
        filename=f"Writing {Path(output.arrow_file).name}",
        start=False,
    )

    tables = []
    for file in files:
        with pa.memory_map(str(file), 'rb') as source:
            iTable = pa.ipc.open_file(source).read_all()
        tables.append(iTable)
    progress.update(task_id, description="concatenating")
    big_table = pa.concat_tables(tables)

    progress.update(task_id, description="sorting")
    sortidx = get_sort_index(big_table, sort)
    progress.update(task_id, total=len(sortidx))
    progress.update(task_id, description="writing")
    progress.start_task(task_id)
    with pa.OSFile(output.arrow_file, 'wb') as sink:
        with pa.ipc.new_file(sink, big_table.schema) as writer:
            for start in range(0,len(sortidx),output.batch_size):
                subset = sortidx.slice(start,output.batch_size)
                batch = big_table.take(subset)
                writer.write(batch)
                progress.update(task_id, advance=len(batch))
                #print(f"Wrote batch {start}:{start+len(batch)}")

@hydra.main(config_path="conf", config_name="config_transform_table", version_base=None)
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))

    with tempfile.TemporaryDirectory(prefix=os.path.join(cfg.temp.dir,'')) as tempdirname:
        files = []
        with progress:
            for filename in cfg.input.files:
                ifile = Path(filename)
                ofile = Path(tempdirname) / ifile.name
                process_and_cache(ifile, ofile, cfg.sort, batch_size=cfg.output.batch_size, casts=cfg.casts, operations=cfg.operations)
                files.append(ofile)
            #global_console.print("All files loaded")
            concat_to_output(files, cfg.sort, cfg.output)
    
    # Create parquet file
    print(f"Creating {cfg.output.parquet_file}...", end='')
    with pa.memory_map(cfg.output.arrow_file, 'rb') as source:
        table = pa.ipc.open_file(source).read_all()
    pq.write_table(table, cfg.output.parquet_file)
    print("done.")

if __name__ == "__main__":
    main()
