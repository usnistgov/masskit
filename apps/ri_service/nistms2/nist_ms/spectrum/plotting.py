import matplotlib.pyplot as plt
import numpy as np


def unsparsify_spectrum(spectrum, max_mz):
    """
    fill out array using spectrum values, placing zeros in missing mz values
    :param spectrum: the spectrum to operate on
    :param max_mz: maximum mz value
    :return: the unsparsified array
    """
    return np.fromiter((spectrum.products.intensity[np.nonzero(spectrum.products.mz == x)[0][0]] if
                        np.nonzero(spectrum.products.mz == x)[0].shape[0] > 0 else 0 for x in range(max_mz)),
                       dtype="float64")


def mirror_plot(query, subject=None, title=None):
    """
    create a spectrum plot.  If subject spectrum is specified, will draw a mirror plot
    :param query: spectrum to be plotted
    :param subject: optional second spectrum for mirror plot
    :param title: title of graph
    """
    ax = plt.subplot(111)

    if title is not None:
        ax.set_title(title)
    if subject is None:
        max_mz = np.max(query.products.mz)
    else:
        max_mz = np.max([np.max(query.products.mz), np.max(subject.products.mz)])

    # create the bar charts
    # check to see if integer mz to set width
    if issubclass(query.products.mz.dtype.type, np.integer):
        width = 1
    else:
        width = max_mz/10000.
    ax.bar(query.products.mz, query.products.intensity, width=width, color='b')
    if subject is not None:
        ax.bar(subject.products.mz, np.negative(subject.products.intensity), width=width, color='r')
        # plot the x axis
        plt.axhline(0, color='black', linewidth=1)
    return

