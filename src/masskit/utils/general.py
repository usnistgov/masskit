import importlib
import os
import logging
from pathlib import Path, PurePosixPath
import tarfile
import numpy as np
import gzip
import bz2
from urllib import request
from urllib.parse import urlparse
try:
    import boto3
except ImportError:
    logging.debug("Unable to import boto3")
    boto3 = None

def class_for_name(module_name_list, class_name):
    """
    dynamically try to find class in list of modules
    :param module_name_list: list of modules to search
    :param class_name: class to look for
    :return: class
    """
    if class_name is None:
        return None
    # load the module, will raise ImportError if module cannot be loaded
    c = None
    for module_name in module_name_list:
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        if m is not None:
            try:
                c = getattr(m, class_name)
            except AttributeError:
                continue
            break
    if c is None:
        raise ImportError(f"unable to find {class_name}")
    return c

def open_if_compressed(filename):
    magic_dict = {
        b"\x1f\x8b\x08": "gzip",
        b"\x42\x5a\x68": "bzip2"
    }

    max_len = max(len(x) for x in magic_dict)

    def get_type(filename):
        with open(filename, "rb") as f:
            file_start = f.read(max_len)
        for magic, filetype in magic_dict.items():
            if file_start.startswith(magic):
                return filetype
        return "text"

    file_type = get_type(filename)

    if file_type == "gzip":
        return gzip.open(filename, "rt")
    elif file_type == "bzip2":
        return bz2.open(filename, "rt")
    else:
        return open(filename)

def open_if_filename(fp, mode, newline=None):
    """
    if the fp is a string, open the file

    :param fp: possible filename
    :param mode: file opening mode
    :return: stream
    """
    if isinstance(fp, str) or isinstance(fp, Path) :
        fp = open(fp, mode, newline=newline)
        if not fp:
            raise ValueError(f"not able to open file")
    return fp


def discounted_cumulative_gain(relevance_array):
    return np.sum((np.power(2.0, relevance_array) - 1.0) / np.log2(np.arange(2, len(relevance_array) + 2)))


def parse_filename(filename: str):
    """
    parse filename into root and extension

    :param filename: filename
    :return: root, extension
    """
    path = Path(filename).expanduser()
    return path.with_suffix(""), path.suffix[1:]


def search_for_file(filename, directories):
    """
    look for a file in a list of directories

    :param filename: the filename to look for
    :param directories: the directories to search
    :return: the Path to the file, otherwise None if not found
    """
    for directory in directories:
        path = Path(directory, filename)
        if path.is_file():
            return path
    return None


def get_file(filename, cache_directory=None, search_path=None, tgz_extension=None):
    """
    for a given file, return the path (downloading file if necessary)

    :param filename: name of the file
    :param cache_directory: where files are cached (config.paths.cache_directory)
    :param search_path: list of where to search for files (config.paths.search_path)
    :param tgz_extension: if a tgz file to be downloaded, the extension of the unarchived file
    :return: path

    Notes: we don't support zip files as the python api for zip files doesn't support symlinks
    """
    if cache_directory is None:
        cache_directory = Path.home() / Path('.masskit_cache')
    else:
        cache_directory = Path(cache_directory).expanduser()  # expand tilde notation
    if search_path is None:
        search_path = [Path.home() / Path('.masskit_cache')]
    if cache_directory not in search_path:
        search_path.append(cache_directory)
    search_path = [Path(x).expanduser() for x in search_path]  # expand tilde notation

    # treat as an url
    url = urlparse(filename, allow_fragments=False)
    is_url = url.scheme in ["s3", "http", "https"]
    url_path = PurePosixPath(url.path)
    is_tgz = tgz_extension is not None and url_path.suffix == '.tgz'

    # treat as a file, get the path
    file_path = Path(filename).expanduser()

    # create non-tgz version of file name
    if is_url:
        if is_tgz:
            final_filename = Path(url_path.with_suffix(tgz_extension).name)
        else:
            final_filename = Path(url_path.name)
    else:
        final_filename = Path(file_path.name)


    # look for the non-tgz version of the file.  If it's there, return the path
    if not is_url and file_path.is_file():
        return file_path
    else:
        file_path = search_for_file(final_filename, search_path)
        if file_path is not None:
            return file_path
        
    # assume we need to download the file
    if is_url:
        cache_directory.mkdir(parents=True, exist_ok=True)  # make the cache directory if necessary
        if url.scheme == "s3":
            s3 = boto3.client("s3")
            with open(cache_directory / url_path.name, "wb") as f:
                s3.download_fileobj(url.netloc, url.path, f)
        else: 
            request.urlretrieve(url.geturl(), cache_directory / url_path.name)

        if is_tgz:
            with tarfile.open(cache_directory / url_path.name, 'r:gz') as tgz_ref:
                tgz_ref.extractall(cache_directory)
            os.remove(cache_directory / url_path.name)
        
        if (cache_directory / final_filename).is_file():
            return cache_directory / final_filename

    return None
