import importlib
import os
import logging
from pathlib import Path
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
    file_extension = os.path.splitext(filename)[1][1:].lower()
    file_root = os.path.splitext(filename)[0]
    if file_extension in ["pickle", "pck", "pcl", "pkl"]:
        file_extension = "pkl"
    return file_root, file_extension


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


def get_file(filename, cache_directory=None, search_path=None):
    """
    for a given file, return the path (downloading file if necessary)

    :param filename: name of the file
    :param cache_directory: where files are cached (config.paths.cache_directory)
    :param search_path: list of where to search for files (config.paths.search_path)
    :return: path
    """
    if cache_directory is None:
        cache_directory = Path.home() / Path('.massspec_cache')
    if search_path is None:
        search_path = Path.home() / Path('.massspec_cache')

    url = urlparse(filename, allow_fragments=False)
    if url.scheme in ["s3", "http", "https"]:
        path = search_for_file(url.path.lstrip("/"), search_path)
        # if the file doesn't exist in the cache, download it
        if path is None or not path.is_file():
            path = Path(cache_directory, url.path.lstrip("/"))
            path.parent.mkdir(
                parents=True, exist_ok=True
            )  # make the cache directory
            if url.scheme == "s3":
                s3 = boto3.client("s3")
                with open(path, "wb") as f:
                    s3.download_fileobj(url.netloc, url.path.lstrip("/"), f)
            else:
                request.urlretrieve(filename, path)
    else:
        path = Path(filename)

    return path

