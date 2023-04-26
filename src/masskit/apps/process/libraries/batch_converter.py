import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig
from masskit.spectrum.spectrum import init_spectrum
from masskit.utils.files import BatchFileReader, BatchFileWriter, load_mgf2array, load_msp2array, spectra_to_msp, spectra_to_mgf
from masskit.utils.general import parse_filename
import pyarrow as pa
import pyarrow.parquet as pq
from masskit.utils.tables import row_view
import os
from typing import Union
from pyarrow import Table
from collections.abc import Iterable

"""
streaming file converter with multiprocessing
takes a variety of inputs, including parquet and arrow formatted files and converts them 
in batches to msp, mgf, arrow, and parquet formatted files.

batch_converter.py --config config_converter
batch_converter.py input.file.names=TestUniqSynPho202249.msp output.file.name=TestUniqSynPho202249.mzxml
"""


@hydra.main(config_path="conf", config_name="config_batch_converter", version_base=None)
def batch_converter_app(config: DictConfig) -> None:

    logging.getLogger().setLevel(logging.INFO)

    if config.conversion.msp.comment_fields:
        comment_fields = eval(config.conversion.msp.comment_fields[0])
    else:
        comment_fields = None


    output_file_root, output_file_extension = parse_filename(Path(config.output.file.name).expanduser())
    if config.output.file.types:
        output_file_extension = config.output.file.types if isinstance(config.output.file.types, Iterable) else\
            [config.output.file.types]
    else:
        output_file_extension = [output_file_extension]

    if not output_file_root:
        raise ValueError("no output file specified")

    input_files = config.input.file.names if not isinstance(config.input.file.names, str) else [config.input.file.names]

    # create the batch writers
    writers = []
    for output_extension in output_file_extension:
        writers.append(BatchFileWriter(output_file_root.with_suffix(f".{output_extension}"), 
                                       format=output_extension, 
                                       annotate=config.conversion.annotate, 
                                       row_batch_size=config.conversion.row_batch_size))
    for input_file in input_files:
        input_file = str(Path(input_file).expanduser())
        # use the file extension to determine file type unless specified in the arguments
        input_file_root, input_file_extension = parse_filename(input_file)
        if config.input.file.type:
            input_file_extension = config.input.file.type
        if (
            input_file_root == output_file_root
            and input_file_extension in output_file_extension
        ):
            raise ValueError("output file will overwrite input file")

        # use a named id field or use an integer initial value for the id
        id = config.conversion[input_file_extension].id
        if id['field']:
            id_field = id['field']
        else:
            id_field = id['initial_value']

        reader = BatchFileReader(input_file, format=input_file_extension,
                                 row_batch_size=config.conversion.row_batch_size,
                                 id_field=id_field,
                                 comment_fields=comment_fields)

        for writer in writers:
            num_rows = 0
            for table in reader.iter_tables():
                if config.input.num is not None and config.input.num > 0 and len(table) + num_rows > config.input.num:
                    table = table.slice(0, config.input.num - num_rows)
                    writer.write_table(table)
                    break
                writer.write_table(table)
                num_rows += len(table)


if __name__ == "__main__":
    batch_converter_app()
