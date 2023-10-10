import math

import imageio
import numpy as np
from matplotlib import collections as mc
from matplotlib import pyplot as plt

from .. import constants as _mkconstants


def unsparsify_spectrum(spectrum, max_mz):
    """
    fill out array using spectrum values, placing zeros in missing mz values

    :param spectrum: the spectrum to operate on
    :param max_mz: maximum mz value
    :return: the unsparsified array
    """
    return np.fromiter(
        (
            spectrum.products.intensity[np.nonzero(spectrum.products.mz == x)[0][0]]
            if np.nonzero(spectrum.products.mz == x)[0].shape[0] > 0
            else 0
            for x in range(max_mz)
        ),
        dtype="float64",
    )


class AnimateSpectrumPlot:
    """
    class used to create animated gifs of spectrum plots
    """

    def __init__(self):
        # list of images for doing animated gif
        self.image_list = []

    def add_figure(self, fig, close_fig=True):
        """
        use the provided figure to draw an image for one frame of an animation

        :param fig: matplotlib figure
        :param close_fig: if True, close the figure when done
        """
        fig.canvas.draw()  # draw the canvas, cache the renderer
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.image_list.append(image)
        if close_fig:
            plt.close(fig)

    def create_animated_gif(self, file, fps=5, pause=7):
        """
        create the animated gif from the stored figures

        :param file: the file to write the animated gif to
        :param fps: how many frames per second the animation runs
        :param pause: how many time to duplicate the last frame
        """
        # kwargs_write = {'fps': 0.2, 'quantizer': 'nq'}
        # duplicate last image
        image_list_final = self.image_list.copy()
        image_list_final.extend([self.image_list[-1] for _ in range(pause)])
        imageio.mimwrite(file, image_list_final, fps=fps, subrectangles=True)



def normalize_intensity(intensity, normalize=999.0):
    """
    norm the spectrum to the max peak

    :param intensity:
    :param normalize: value to norm the spectrum to
    :return:
    """
    divisor = intensity.max()
    return intensity * normalize / (divisor + _mkconstants.EPSILON)


def line_plot(mz_in, intensity_in, color, linewidth=1):
    """
    create a LineCollection for plotting a spectrum

    :param mz_in: mz values
    :param intensity_in: intensity value
    :param color: color of spectrum
    :return: line collection
    """
    lines = []
    intensity_nonzero = intensity_in != 0.0
    for mz_val, intensity_val in zip(
        mz_in[intensity_nonzero], intensity_in[intensity_nonzero]
    ):
        lines.append([(mz_val, intensity_val), (mz_val, 0)])
    lc = mc.LineCollection(lines, colors=color, linewidths=linewidth)
    return lc


def error_bar_plot(mz_in, intensity_in, stddev_in, color, linewidth=1):
    """
    plot spectra as colored error bars

    :param mz_in: mz values in daltons
    :param intensity_in: intensity value
    :param stddev_in: standard deviations from intensity_in
    :param color: color of spectrum
    :param alpha: alpha blending
    :return: line collection
    """
    vertices = []
    intensity_nonzero = (intensity_in != 0.0) | (stddev_in != 0.0)
    for mz_val, intensity_val, stddev_val in zip(
        mz_in[intensity_nonzero],
        intensity_in[intensity_nonzero],
        stddev_in[intensity_nonzero],
    ):
        vertices.append(
            [
                (mz_val - 3, intensity_val - stddev_val / 2.0),
                (mz_val - 3, intensity_val + stddev_val / 2.0),
                (mz_val + 8, intensity_val + stddev_val / 2.0),
                (mz_val + 8, intensity_val - stddev_val / 2.0),
            ]
        )
    pc = mc.PolyCollection(vertices, facecolors=color, zorder=-1, linewidth=linewidth)
    return pc


def error_bar_plot_lines(mz_in, intensity_in, stddev_in, color, vertical_cutoff=0.01, linewidth=1):
    """
    plot spectra as colored error bars using lines

    :param mz_in: mz values in daltons
    :param intensity_in: intensity value
    :param stddev_in: standard deviations from intensity_in
    :param color: color of spectrum
    :param vertical_cutoff: if the intensity/max_intensity is below this value, don't plot the vertical line
    :return: line collection
    """
    lines = []
    max_intensity = np.max(intensity_in)
    intensity_nonzero = (intensity_in != 0.0) | (stddev_in != 0.0)
    for mz_val, intensity_val, stddev_val in zip(
        mz_in[intensity_nonzero],
        intensity_in[intensity_nonzero],
        stddev_in[intensity_nonzero],
    ):
        if abs(intensity_val) >= abs(vertical_cutoff * max_intensity):
            vertical_length = max(stddev_val,  0.01 * abs(max_intensity))
            lines.append(
                [
                    (mz_val, intensity_val + vertical_length),
                    (mz_val, intensity_val - vertical_length),
                ]
            )
            lines.append([(mz_val - 7, intensity_val), (mz_val + 7, intensity_val)])
            lines.append(
                [
                    (mz_val - 6, intensity_val + stddev_val),
                    (mz_val + 6, intensity_val + stddev_val),
                ]
            )
            lines.append(
                [
                    (mz_val - 6, intensity_val - stddev_val),
                    (mz_val + 6, intensity_val - stddev_val),
                ]
            )
    lc = mc.LineCollection(lines, colors=color, linewidths=linewidth, zorder=-1)
    return lc


def spectrum_plot(
    axis,
    mz,
    intensity,
    stddev=None,
    mirror_mz=None,
    mirror_intensity=None,
    mirror_stddev=None,
    mirror=True,
    title=None,
    xlabel='m/z',
    ylabel='Intensity',
    title_size=None,
    label_size=None,
    max_mz=None,
    min_mz=0,
    color=(0, 0, 1, 1),
    mirror_color=(1, 0, 0, 1),
    stddev_color=(0.3, 0.3, 0.3, 0.5),
    left_label_color=(1, 0, 0, 1),
    normalize=1000,
    vertical_cutoff=0.0,
    vertical_multiplier=1.1,
    right_label=None,
    left_label=None,
    right_label_size=None,
    left_label_size=None,
    no_xticks=False,
    no_yticks=False,
    linewidth=1
):
    """
    make a spectrum plot using matplotlib.  if mirror_intensity is specified, will do a mirror plot

    :param axis: matplotlib axis
    :param mz: mz values as array-like
    :param intensity: intensity as array-like, parallel to mz array
    :param stddev: standard deviation of the intensities
    :param title: title of plot
    :param xlabel: xlabel of plot
    :param ylabel: ylabel of plot
    :param title_size: size of title font
    :param label_size: size of x and y axis label fonts
    :param mirror_mz: mz values of mirror spectrum, corresponding to mirror_intensity.  If none, uses mz
    :param mirror_intensity: intensity of mirror spectrum as array-like, parallel to mz array.  If none, don't plot
    :param mirror_stddev: standard deviation of the intensities
    :param mirror: if true, mirror the plot if there are two spectra.  Otherwise plot the two spectra together
    :param max_mz: maximum mz to plot
    :param min_mz: minimum mz to plot
    :param normalize: if specified, norm the spectra to this value.
    :param color: color of spectrum specified as RBGA tuple
    :param mirror_color: color of mirrored spectrum specified as RGBA tuple
    :param stddev_color: color of error bars
    :param left_label_color: color of the left top label
    :param vertical_cutoff: if the intensity/max_intensity is below this value, don't plot the vertical line
    :param vertical_multiplier: multiply times y max values to create white space
    :param right_label: label for the top right of the fiture
    :param left_label: label for the top left of the figure
    :param right_label_size: size of label for the top right of the fiture
    :param left_label_size: size of label for the top left of the figure
    :param no_xticks: turn off x ticks and labels
    :param no_yticks: turn off y ticks and lables
    :param linewidth: width of plotted lines
    :return: peak_collection, mirror_peak_collection sets of peaks for picking
    """

    if axis is None:
        axis = plt.gca()

    def finalize_plot(axis, max_mz, min_mz, vertical_multiplier, right_label, left_label, right_label_size, left_label_size, y_lim_lo, y_lim, left_label_color):
        axis.set_ylim([y_lim_lo*vertical_multiplier, y_lim*vertical_multiplier])
        axis.set_xlim([min_mz, max_mz])
        if left_label is not None:
            axis.text(0.02, 0.95, left_label, horizontalalignment='left', verticalalignment='top', transform=axis.transAxes, size=left_label_size, color=left_label_color)
        if right_label is not None:
            axis.text(0.98, 0.95, right_label, horizontalalignment='right', verticalalignment='top', transform=axis.transAxes, size=right_label_size, color=left_label_color)


    # line collections of the peaks, returned for picking
    peak_collection = None
    mirror_peak_collection = None

    if xlabel is None:
        xlabel='m/z'
    if xlabel is None:
        ylabel='Intensity'
    
    axis.set_xlabel(xlabel, fontsize=label_size)
    axis.set_ylabel(ylabel, fontsize=label_size)

    if no_xticks:
        axis.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axis.set_xlabel(None, fontsize=label_size)

    if no_yticks:
        axis.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  
        axis.set_ylabel(None, fontsize=label_size)      

    if len(mz) == 0 or len(intensity) == 0:
        return peak_collection, mirror_peak_collection
    
    if title is not None:
        axis.set_title(title, fontsize=title_size)

    y_lim_lo = 0

    # create the bar charts
    if normalize:
        intensity = normalize_intensity(intensity, normalize)

    # find max intensity value for y_lim
    if max_mz is not None and min_mz is not None:
        y_lim = max(intensity[(mz >= min_mz) & (mz <= max_mz)])
    else:
        y_lim = max(intensity)

    if stddev is not None:
        axis.add_collection(
            error_bar_plot_lines(
                mz, intensity, stddev, stddev_color, vertical_cutoff=vertical_cutoff, linewidth=linewidth
            )
        )
    peak_collection = line_plot(mz, intensity, color, linewidth=linewidth)
    axis.add_collection(peak_collection)

    if mirror_mz is None or mirror_intensity is None or len(mirror_mz) == 0 or len(mirror_intensity) == 0:
        if max_mz is None:
            max_mz = max(mz)
        if min_mz is None:
            min_mz = min(mz)
        finalize_plot(axis, max_mz, min_mz, vertical_multiplier, right_label, left_label, right_label_size, left_label_size, y_lim_lo, y_lim, left_label_color)
        return peak_collection, mirror_peak_collection

    if mirror_mz is None:
        mirror_mz = mz

    if max_mz is None:
        max_mz = max(max(mirror_mz), max(mz))
    if min_mz is None:
        min_mz = min(min(mirror_mz), min(mz))

    if normalize:
        mirror_intensity = normalize_intensity(mirror_intensity, normalize)

    if max_mz is not None and min_mz is not None:
        y_lim = max(y_lim, max(mirror_intensity[(mirror_mz >= min_mz) & (mirror_mz <= max_mz)]))
    else:
        y_lim = max(y_lim, max(mirror_intensity))

    if mirror:
        mirror_intensity = -mirror_intensity
        y_lim_lo = -y_lim
        mirror_peak_collection = line_plot(mirror_mz, mirror_intensity, mirror_color, linewidth=linewidth)
        axis.add_collection(mirror_peak_collection)
        axis.axhline(0, color="black", linewidth=linewidth)
    if mirror_stddev is not None:
        axis.add_collection(
            error_bar_plot_lines(
                mirror_mz,
                mirror_intensity,
                mirror_stddev,
                stddev_color,
                vertical_cutoff=vertical_cutoff,
                linewidth=linewidth
            )
        )

    finalize_plot(axis, max_mz, min_mz, vertical_multiplier, right_label, left_label, right_label_size, left_label_size, y_lim_lo, y_lim, left_label_color)
    return peak_collection, mirror_peak_collection


def multiple_spectrum_plot(
    intensities,
    mz=None,
    mirror_intensities=None,
    dpi=100,
    min_mz=0,
    max_mz=2000,
    title="",
    subtitles=None,
    normalize=None,
    color=(0, 0, 1, 1),
    mirror_color=(1, 0, 0, 1),
):
    """
    create a spectrum plot.  If subject spectrum is specified, will draw a mirror plot

    :param intensities: spectrum to be plotted.  array-like
    :param mz: mz values for plot. array-like parallel to intensities
    :param mirror_intensities: intensities for mirror spectrum. array-like parallel to intensities.
    :param dpi: dpi of the plot
    :param min_mz: minimum mz of the plot
    :param max_mz: maximum mz of the plot
    :param title: title of the plot
    :param subtitles: an array of strings, one title for each spectrum plot
    :param normalize: norm the spectra intensities to this value
    :param color: color of spectrum specified as RBGA tuple
    :param mirror_color: color of mirrored spectrum specified as RGBA tuple
    :return: matplotlib figure
    """
    plt.ioff()
    figure_size = math.ceil(math.sqrt(intensities.shape[0]))
    fig, ax = plt.subplots(
        figure_size,
        figure_size,
        figsize=(figure_size * 2, figure_size * 2),
        sharey="row",
        dpi=dpi,
    )
    i = 0  # iterator through spectra

    # iterate through the subplots
    for row in ax:
        for col in row:
            if subtitles is not None:
                subtitle = subtitles[i]
            else:
                subtitle = None
            if mirror_intensities is not None:
                mirror_intensity = mirror_intensities[i]
            else:
                mirror_intensity = None
            spectrum_plot(
                col,
                mz,
                intensities[i],
                mirror_intensity=mirror_intensity,
                title=subtitle,
                xlabel=None,
                ylabel=None,
                title_size=8,
                label_size=8,
                max_mz=max_mz,
                min_mz=min_mz,
                color=color,
                mirror_color=mirror_color,
                normalize=normalize,
            )
            i += 1
            if (
                i == intensities.shape[0]
            ):  # if at the end of the list, break out of the two loops
                break
        else:
            continue
        break
    fig.suptitle(title, fontsize=16)
    # adjust the figure so there are no overlaps
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

def draw_spectrum(spectrum, fig_format, output, figsize=(4, 2)):
    """
    spectrum thumbnail plotting code called by the spectrum object
    writes to a stream
    
    :param spectrum: spectrum to plot
    :param fig_format: format of the figure e.g. 'png'
    :param output: output stream
    :param figsize: the size of the figure in inches
    """

    plt.ioff()  # turn off interactive mode
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    spectrum.plot(ax)
    plt.tight_layout()
    fig.savefig(output, format=fig_format)
    plt.close(fig)
    plt.ion()  # turn on interactive mode
    return output.getvalue()