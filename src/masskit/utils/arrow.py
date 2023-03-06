import os
from pathlib import Path
import tempfile
from masskit.utils.index import ArrowLibraryMap
import hashlib
import pyarrow as pa
from filelock import FileLock

def create_object_id(filename, filters):
    """
    create an object id based on filename and filters

    :param filename: filename
    :param filters: filter string
    :return: object id
    """
    object_name = str(filename)
    if filters is not None:
        object_name += str(filters)
    return hashlib.sha1(object_name.encode('utf-8'))

def save_to_arrow(filename, columns, filters):
    """
    Load a parquet file and save it as a temp arrow file after applying
    filters and column lists.  Load it as a memmap if the temp arrow file
    already exists

    :param filename: parquet file
    :param columns: columns to load
    :param filters: parquet file filters
    :return: ArrowLibraryMap
    """
    obj_id = create_object_id(filename, filters).hexdigest() + '.arrow'
    temp_file_path = Path(tempfile.gettempdir()) / obj_id
    data = None

    # need to use a file lock here as the file might be partially written
    # note that on linux, the lock file persists after the process exits because of
    # https://stackoverflow.com/questions/58098634/why-does-the-python-filelock-library-delete-lockfiles-on-windows-but-not-unix
    with FileLock(temp_file_path.with_suffix(".lock")):
        if temp_file_path.is_file() and os.path.getmtime(filename) < os.path.getmtime(temp_file_path):
            # read in arrow file as memory map
            data = pa.ipc.RecordBatchFileReader(pa.memory_map(str(temp_file_path), 'r')).read_all()
            # create ArrowLibraryMap
            data = ArrowLibraryMap(data)
        else:
            # read from parquet file
            data = ArrowLibraryMap.from_parquet(filename, columns=columns, filters=filters)
            with pa.OSFile(str(temp_file_path), 'wb') as sink:
                with pa.RecordBatchFileWriter(sink, data.to_arrow().schema) as writer:
                    writer.write_table(data.to_arrow())

    return data
