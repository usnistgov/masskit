#!/usr/bin/env python
import logging
from pathlib import Path

import hydra
import pyarrow as pa
from hydra.core.plugins import Plugins
from omegaconf import DictConfig

from masskit.data_specs.spectral_library import *
from masskit.utils.general import MassKitSearchPathPlugin, parse_filename
from masskit.utils.tablemap import ArrowLibraryMap

"""
takes a variety of inputs, including parquet, mgf and msp formatted files and converts them to
msp, sdf and parquet formatted files.

converter.py --config[-path|-name|-dir] config_converter
converter.py input.file.names=TestUniqSynPho202249.msp output.file.name=TestUniqSynPho202249.mzxml
"""


def disable_console_logging():
    log = logging.getLogger()
    xhdl = None
    for hdl in log.handlers:
        if hdl.name == "console":
            xhdl = hdl
    if xhdl:
        log.removeHandler(xhdl)


Plugins.instance().register(MassKitSearchPathPlugin)


@hydra.main(config_path="conf", config_name="config_converter", version_base=None)
def converter_app(config: DictConfig) -> None:

    disable_console_logging()
    logging.getLogger().setLevel(logging.INFO)

    if config.conversion.msp.comment_fields:
        comment_fields = eval(config.conversion.msp.comment_fields[0])
    else:
        comment_fields = None

    output_file_root, output_file_extension, compression = parse_filename(
        Path(config.output.file.name).expanduser())
    if config.output.file.types:
        output_file_extension = config.output.file.types if type(config.output.file.types) is list else\
            [config.output.file.types]
    else:
        output_file_extension = [output_file_extension]

    if not output_file_root:
        raise ValueError("no output file specified")

    tables = []

    input_files = config.input.file.names if not isinstance(
        config.input.file.names, str) else [config.input.file.names]

    id_field = config.conversion.id.field

    for input_file in input_files:
        input_file = str(Path(input_file).expanduser())
        # use the file extension to determine file type unless specified in the arguments
        input_file_root, input_file_extension, compression = parse_filename(
            input_file)
        if config.input.file.type:
            input_file_extension = config.input.file.type
        if (
            input_file_root == output_file_root
            and input_file_extension in output_file_extension
        ):
            raise ValueError("output file will overwrite input file")

        # load in the files
        if input_file_extension == "sdf":
            table = ArrowLibraryMap.from_sdf(
                input_file,
                max_size=config.conversion.small_molecule.max_bound,
                num=config.input.num,
                source=config.input.file.source,
                id_field=id_field,
                min_intensity=config.conversion.spectra.min_intensity,
                set_probabilities=config.conversion.set.probabilities,
            )
        elif input_file_extension == "msp":
            table = ArrowLibraryMap.from_msp(
                input_file,
                num=config.input.num,
                id_field=id_field,
                comment_fields=comment_fields,
                min_intensity=config.conversion.spectra.min_intensity,
            )
        elif input_file_extension == "mgf":
            table = ArrowLibraryMap.from_mgf(
                input_file,
                num=config.input.num,
                min_intensity=config.conversion.spectra.min_intensity,
            )
        elif input_file_extension == "parquet":
            table = ArrowLibraryMap.from_parquet(
                input_file,
                num=config.input.num,
            )
        else:
            raise NotImplementedError

        tables.append(table)
        if type(id_field) is int:
            id_field += len(table)

    library_map = ArrowLibraryMap(pa.concat_tables(
        [table.to_arrow() for table in tables], promote=True), num=config.input.num)

    # output the files
    for output_extension in output_file_extension:
        if output_extension == "mzxml":
            library_map.to_mzxml(output_file_root.with_suffix(".mzxml"))
        elif output_extension == "parquet":
            library_map.to_parquet(output_file_root.with_suffix(".parquet"))
        elif output_extension == "msp":
            library_map.to_msp(output_file_root.with_suffix(".msp"))
        elif output_extension == "mgf":
            library_map.to_mgf(output_file_root.with_suffix(".mgf"))


if __name__ == "__main__":
    converter_app()
