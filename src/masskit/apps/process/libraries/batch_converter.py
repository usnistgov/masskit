#!/usr/bin/env python
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig, ListConfig
from masskit.utils.files import BatchFileReader, BatchFileWriter
from masskit.utils.general import MassKitSearchPathPlugin, expand_path_list, parse_filename
from hydra.core.plugins import Plugins
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    SpinnerColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.console import Group
from rich.panel import Panel

Plugins.instance().register(MassKitSearchPathPlugin)


overall_progress = Progress(
    TextColumn(
        "[blue]{task.description} {task.fields[filename]}: ", justify="right"),
    TimeElapsedColumn(), MofNCompleteColumn()
)

write_progress = Progress(
    TextColumn(
        "[blue]{task.description} {task.fields[filename]}: ", justify="right"),
    TimeElapsedColumn(), MofNCompleteColumn()
)

batch_progress = Progress(
    TextColumn(
        "[blue]{task.description}: {task.completed}", justify="right"),
    SpinnerColumn("simpleDots")
)

progress_group = Group(
    overall_progress, write_progress, batch_progress,
)

# logging.basicConfig(level="NOTSET", handlers=[RichHandler(level="NOTSET")])
# logger = logging.getLogger('rich')


@hydra.main(config_path="conf", config_name="config_batch_converter", version_base=None)
def batch_converter_app(config: DictConfig) -> None:

    output_file_root, output_file_extension, compression = parse_filename(
        Path(config.output.file.name).expanduser())
    if config.output.file.types:
        output_file_extension = config.output.file.types if isinstance(config.output.file.types, list) or \
            isinstance(config.output.file.types, ListConfig) else [config.output.file.types]
    else:
        output_file_extension = [output_file_extension]

    if not output_file_root:
        raise ValueError("no output file specified")

    input_files = expand_path_list(config.input.file.names)

    with Live(progress_group):
        # create the batch writers
        writers = []
        for output_extension in output_file_extension:
            writers.append(BatchFileWriter(output_file_root.with_suffix(f".{output_extension}"),
                                           format=output_extension,
                                           annotate=config.conversion.get(
                "annotate", False),
                row_batch_size=config.conversion.get("row_batch_size", 5000)))
        overall_task_id = overall_progress.add_task(
            "Load", total=len(input_files))
        for input_file in input_files:
            input_file = str(input_file)
            overall_progress.update(
                overall_task_id, description="Load", filename=input_file)
            # use the file extension to determine file type unless specified in the arguments
            input_file_root, input_file_extension, compression = parse_filename(
                input_file)
            if (
                input_file_root == output_file_root
                and input_file_extension in output_file_extension
            ):
                raise ValueError("output file will overwrite input file")

            format = dict(config.conversion.get(
                input_file_extension.lower(), None))

            reader = BatchFileReader(input_file,
                                     format=format,
                                     row_batch_size=config.conversion.get(
                                         "row_batch_size", 5000),
                                     )
            writer_task_id = write_progress.add_task(
                "Write", total=len(writers))
            for writer in writers:
                write_progress.update(
                    writer_task_id, description="Write", filename=writer.filename)
                num_rows = 0
                batch_task_id = batch_progress.add_task(" Write batch")
                for table in reader.iter_tables():
                    batch_progress.update(
                        batch_task_id, description=" Write batch")
                    if config.input.num is not None and config.input.num > 0 and len(table) + num_rows > config.input.num:
                        table = table.slice(0, config.input.num - num_rows)
                        writer.write_table(table)
                        batch_progress.update(batch_task_id, advance=1)
                        break
                    writer.write_table(table)
                    num_rows += len(table)
                    batch_progress.update(batch_task_id, advance=1)
                write_progress.update(writer_task_id, advance=1)
            overall_progress.update(overall_task_id, advance=1)
        for writer in writers:
            writer.close()


if __name__ == "__main__":
    batch_converter_app()
