import importlib
import os
from pathlib import Path

import numpy as np


def class_for_name(module_name_list, class_name):
    """
    dynamically try to find class in list of modules
    :param module_name_list: list of modules to search
    :param class_name: class to look for
    :return: class
    """
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


def open_if_filename(fp, mode):
    """
    if the fp is a string, open the file

    :param fp: possible filename
    :param mode: file opening mode
    :return: stream
    """
    if isinstance(fp, str) or isinstance(fp, Path) :
        fp = open(fp, mode)
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