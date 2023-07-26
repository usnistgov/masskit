import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, ListConfig
from masskit.utils.files import BatchFileReader, BatchFileWriter
from masskit.utils.general import expand_path_list, parse_filename


@hydra.main(config_path="conf", config_name="config_batch_converter", version_base=None)
def batch_converter_app(config: DictConfig) -> None:

    logging.getLogger().setLevel(logging.INFO)

    output_file_root, output_file_extension, compression = parse_filename(Path(config.output.file.name).expanduser())
    if config.output.file.types:
        output_file_extension = config.output.file.types if isinstance(config.output.file.types, list) or \
             isinstance(config.output.file.types, ListConfig) else [config.output.file.types]
    else:
        output_file_extension = [output_file_extension]

    if not output_file_root:
        raise ValueError("no output file specified")

    input_files = expand_path_list(config.input.file.names)

    # create the batch writers
    writers = []
    for output_extension in output_file_extension:
        writers.append(BatchFileWriter(output_file_root.with_suffix(f".{output_extension}"), 
                                       format=output_extension, 
                                       annotate=config.conversion.get("annotate", False), 
                                       row_batch_size=config.conversion.get("row_batch_size", 5000)))
    for input_file in input_files:
        input_file = str(input_file)
        # use the file extension to determine file type unless specified in the arguments
        input_file_root, input_file_extension, compression = parse_filename(input_file)
        if (
            input_file_root == output_file_root
            and input_file_extension in output_file_extension
        ):
            raise ValueError("output file will overwrite input file")

        format = dict(config.conversion.get(input_file_extension.lower(), None))

        reader = BatchFileReader(input_file, 
                                 format=format,
                                 row_batch_size=config.conversion.get("row_batch_size", 5000),
                                 )

        for writer in writers:
            num_rows = 0
            for table in reader.iter_tables():
                if config.input.num is not None and config.input.num > 0 and len(table) + num_rows > config.input.num:
                    table = table.slice(0, config.input.num - num_rows)
                    writer.write_table(table)
                    break
                writer.write_table(table)
                num_rows += len(table)
            writer.close()


if __name__ == "__main__":
    batch_converter_app()
