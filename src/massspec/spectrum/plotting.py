import math
import imageio
import numpy as np
from massspec_ml.pytorch.spectrum.peptide.peptide_constants import EPSILON
from matplotlib import collections as mc, pyplot as plt


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


def normalize_intensity(intensity, normalize=1.0):
    """
    norm the spectrum to the max peak

    :param intensity:
    :param normalize: value to norm the spectrum to
    :return:
    """
    divisor = intensity.max()
    return intensity * normalize / (divisor + EPSILON)


def line_plot(mz_in, intensity_in, color):
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
    lc = mc.LineCollection(lines, colors=color, linewidths=1)
    return lc


def error_bar_plot(mz_in, intensity_in, stddev_in, color):
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
    pc = mc.PolyCollection(vertices, facecolors=color, zorder=-1)
    return pc


def error_bar_plot_lines(mz_in, intensity_in, stddev_in, color, vertical_cutoff=0.01):
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
    lc = mc.LineCollection(lines, colors=color, linewidths=1, zorder=-1)
    return lc


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
    max_mz=2000,
    min_mz=0,
    color=(0, 0, 1, 1),
    mirror_color=(1, 0, 0, 1),
    stddev_color=(0.3, 0.3, 0.3, 0.5),
    normalize=None,
    vertical_cutoff=0.0,
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
    :param vertical_cutoff: if the intensity/max_intensity is below this value, don't plot the vertical line
    """
    if title:
        if title_size is not None:
            axis.set_title(title, fontsize=title_size)
        else:
            axis.set_title(title)
    if label_size is not None:
        if xlabel:
            axis.set_xlabel(xlabel, fontsize=label_size)
        if ylabel:
            axis.set_ylabel(ylabel, fontsize=label_size)
    else:
        if xlabel:
            axis.set_xlabel(xlabel)
        if ylabel:
            axis.set_ylabel(ylabel)

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
                mz, intensity, stddev, stddev_color, vertical_cutoff=vertical_cutoff
            )
        )
    axis.add_collection(line_plot(mz, intensity, color))

    if mirror_mz is None:
        mirror_mz = mz

    if max_mz is None:
        max_mz = max(max(mirror_mz), max(mz))
    if min_mz is None:
        min_mz = min(min(mirror_mz), min(mz))

    y_lim_lo = 0

    if mirror_intensity is not None:
        if normalize:
            mirror_intensity = normalize_intensity(mirror_intensity, normalize)

        if max_mz is not None and min_mz is not None:
            y_lim = max(y_lim, max(mirror_intensity[(mirror_mz >= min_mz) & (mirror_mz <= max_mz)]))
        else:
            y_lim = max(y_lim, max(mirror_intensity))

        if mirror:
            mirror_intensity = -mirror_intensity
            y_lim_lo = -y_lim
            axis.axhline(0, color="black", linewidth=1)
            axis.add_collection(line_plot(mirror_mz, mirror_intensity, mirror_color))
        if mirror_stddev is not None:
            axis.add_collection(
                error_bar_plot_lines(
                    mirror_mz,
                    mirror_intensity,
                    mirror_stddev,
                    stddev_color,
                    vertical_cutoff=vertical_cutoff,
                )
            )

    axis.set_ylim([y_lim_lo, y_lim])
    axis.set_xlim([min_mz, max_mz])
