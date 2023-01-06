import builtins
from pathlib import Path
import tempfile
from masskit.utils.index import ArrowLibraryMap
from pyarrow.plasma import ObjectID
import hashlib
import pyarrow as pa
from filelock import FileLock


def create_object_id(filename, filters):
    """
    create an object id based on filename and filters
    Note that plasma client must be created and held outside of this function as
    objects in the plasma store are refcounted by connection

    :param filename: filename
    :param filters: filter string
    :return: object id
    """
    object_name = str(filename)
    if filters is not None:
        object_name += str(filters)
    return hashlib.sha1(object_name.encode('utf-8'))

def create_lockfile_name(prefix):
    # use the plasma server pid to create a lockfile.  we use the server pid to avoid race conditions
    tempdir = tempfile.gettempdir()

    if "instance_settings" in dir(builtins) and 'plasma' in builtins.instance_settings and 'pid' in builtins.instance_settings['plasma']:
        lock_file = Path(tempdir, f"{prefix}_{builtins.instance_settings['plasma']['pid']}")
    else:
        lock_file = Path(tempdir, prefix)
    return lock_file

def save_to_plasma(client, filename, columns, filters):
    """
    if parquet file is not in plasma, load as an arrow Table and save it to plasma
    Note that plasma client must be created and held outside of this function as
    objects in the plasma store are refcounted by connection

    :param client: plasma client
    :param filename: parquet file
    :param columns: columns to load
    :param filters: parquet file filters
    :return: ArrowLibraryMap or None if data is already in plasma
    """
    obj_id = ObjectID(create_object_id(filename, filters).digest())
    data = None
        
    lock_file = create_lockfile_name("plasma_lock_save")       

    # check to see if plasma contains the file
    with FileLock(lock_file):
        if not client.contains(obj_id):
            # evaluate the where clause if present
            # load the arrow table from parquet
            data = ArrowLibraryMap.from_parquet(filename, columns=columns, filters=filters)
            
            table = data.to_arrow()
            # save the table to plasma.  we don't use put/get as that is for python objects and doesn't allow for share memory
            # rather, we use an ipc stream, which represents the arrow table as a stream exactly like it is in memory
            buf = client.create(obj_id, table.nbytes)
            stream = pa.FixedSizeBufferWriter(buf)
            with pa.RecordBatchStreamWriter(stream, table.schema) as stream_writer:
                stream_writer.write_table(table)
            client.seal(obj_id)
        else:
            # important to load in data since if this function is called twice in a row, e.g. search a dataset against itself,
            # it needs to return a valid pointer instead of None
            data = load_from_plasma(client, filename, filters)
        
    return data

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
    file_path = Path(tempfile.gettempdir()) / obj_id
    data = None

    # need to use a file lock here as the file might be partially written
    # note that on linux, the lock file persists after the process exits because of
    # https://stackoverflow.com/questions/58098634/why-does-the-python-filelock-library-delete-lockfiles-on-windows-but-not-unix
    with FileLock(file_path.with_suffix(".lock")):
        if file_path.is_file():
            # read in arrow file as memory map
            data = pa.ipc.RecordBatchFileReader(pa.memory_map(str(file_path), 'r')).read_all()
            # create ArrowLibraryMap
            data = ArrowLibraryMap(data)
        else:
            # read from parquet file
            data = ArrowLibraryMap.from_parquet(filename, columns=columns, filters=filters)
            with pa.OSFile(str(file_path), 'wb') as sink:
                with pa.RecordBatchFileWriter(sink, data.to_arrow().schema) as writer:
                    writer.write_table(data.to_arrow())

    return data


def load_from_plasma(client, filename, filters):
    """
    load arrow table from plasma

    :param client: plasma client
    :param filename: name of parquet file
    :param filters: filters on parquet file
    :raises ValueError: can't find the data
    :return: an ArrowLibraryMap
    """
    obj_id = ObjectID(create_object_id(filename, filters).digest())

    lock_file = create_lockfile_name("plasma_lock_load")       

    with FileLock(lock_file):
        if client.contains(obj_id):
            [data] = client.get_buffers([obj_id])
            buf = pa.BufferReader(data)
            with pa.ipc.RecordBatchStreamReader(buf) as stream_reader:
                table = stream_reader.read_all()
                return ArrowLibraryMap(table)
        else:
            raise ValueError('dataset data not in plasma store')
