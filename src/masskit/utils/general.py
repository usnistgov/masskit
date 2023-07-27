import importlib
import os
import logging
from pathlib import Path, PurePosixPath
import tarfile
from typing import Iterable
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
from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin


class MassKitSearchPathPlugin(SearchPathPlugin):
    """
    add the cwd to the search path for configuration yaml files
    """

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        # Appends the search path for this plugin to the end of the search path
        # Note that foobar/conf is outside of the example plugin module.
        # There is no requirement for it to be packaged with the plugin, it just needs
        # be available in a package.
        # Remember to verify the config is packaged properly (build sdist and look inside,
        # and verify MANIFEST.in is correct).
        search_path.prepend(
            provider="masskit-searchpath-plugin", path="."
        )


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


def open_if_compressed(filename, mode, newline=None):
    """
    open the file denoted by filename and uncompress it if it is compressed

    :param filename: filename
    :param mode: file opening mode
    :param newline: specify newline character
    :return: stream
    """
    magic_dict = {
        b"\x1f\x8b\x08": "gzip",
        b"\x42\x5a\x68": "bzip2"
    }

    max_len = max(len(x) for x in magic_dict)

    def get_type(filename):
        if Path(filename).is_file():
            with open(filename, "rb") as f:
                file_start = f.read(max_len)
            for magic, filetype in magic_dict.items():
                if file_start.startswith(magic):
                    return filetype
        return "uncompressed"

    file_type = get_type(filename)

    if file_type == "gzip":
        return gzip.open(filename, mode, newline=newline)
    elif file_type == "bzip2":
        fp = bz2.open(filename, mode, newline=newline)
        # For some reason bz2 does not provide this info automatically
        fp.name = filename
        return fp
    else:
        return open(filename, mode, newline=newline)


def open_if_filename(fp, mode, newline=None):
    """
    if the fp is a string, open the file, and uncompress if needed

    :param fp: possible filename
    :param mode: file opening mode
    :param newline: specify newline character
    :return: stream
    """
    if isinstance(fp, str) or isinstance(fp, Path):
        fp = open_if_compressed(fp, mode, newline=newline)
        if not fp:
            raise ValueError(f"not able to open file")
    return fp


def discounted_cumulative_gain(relevance_array):
    return np.sum((np.power(2.0, relevance_array) - 1.0) / np.log2(np.arange(2, len(relevance_array) + 2)))


def expand_path(path_pattern) -> Iterable[Path]:
    """
    expand a path with globs (wildcards) into a generator of Paths

    :param path_pattern: the path to be globbed
    :return: iterable of Path
    """
    p = Path(path_pattern).expanduser()
    parts = p.parts[p.is_absolute():]
    return Path(p.root).glob(str(Path(*parts)))


def expand_path_list(path_list):
    """
    given a file path or list of file paths that may contain wildcards,
    expand ~ to the user directory and glob the wildcards

    :param path_list: list or str of paths, may include ~ and *
    :return: list of expanded paths
    """
    path_list = path_list if not isinstance(path_list, str) else [path_list]
    return_val = []
    for path in path_list:
        paths = list(expand_path(path))
        return_val.extend(paths)
    return return_val


def parse_filename(filename: str):
    """
    parse filename into root, extension, and compression extension

    :param filename: filename
    :return: root, extension, compression
    """
    path = Path(filename).expanduser()
    root = path.with_suffix("")
    suffix = path.suffix
    if suffix in ['.gz', '.bzip2', '.bz2']:
        if suffix == '.bzip2':
            suffix = '.bz2'
        return root.with_suffix(""), root.suffix[1:], suffix[1:]
    else:
        return root, suffix[1:], ''


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
        # expand tilde notation
        cache_directory = Path(cache_directory).expanduser()
    if search_path is None:
        search_path = [Path.home() / Path('.masskit_cache')]
    if cache_directory not in search_path:
        search_path.append(cache_directory)
    search_path = [Path(x).expanduser()
                   for x in search_path]  # expand tilde notation

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
        # make the cache directory if necessary
        cache_directory.mkdir(parents=True, exist_ok=True)
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
