from matplotlib.collections import PolyCollection
from fasma.core import spectrum as sp
from multiprocessing import Pool
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


color_cycle = {"Fulvous": '#E08217', "Chocolate cosmos": '#67001f', "Tiffany Blue": '#80cdc1',
                  "African Violet": '#9970ab', "Berkekley Blue": '#053061', "Lavender pink": '#F1B6DA',
                  "Dark spring green": '#1B7837', "Ecru": '#DFC27D', "Sky blue": '#92c5de', "Eminence": '#762a83',
                  "Kelly green": '#7FBC41', "Jasper": '#d6604d', "Dark cyan": '#35978F', "Murrey": '#8E0152',
                  "Brown": '#8C510A', "Celadon": '#A6DBA0', "Auburn": '#b2182b', "Green blue": '#2166ac',
                  "Lilac": '#c2a5cf', "Caribbean Current": '#01665E', "Magenta dye": '#C51B7D'}


def plot_sticks(x, y, ax, color, y_min, z=None):
    for root, val in zip(x, y):
        if z is None:
            ax.plot((root, root), (0, val), color=color)
        else:
            ax.plot((root, root), (z, z), (max(y_min, 0), val), color=color)


def plot_line(ax, freq, spect, z=None, **kwargs):
    if z is None:
        ax.plot(freq, spect, **kwargs)
    else:
        poly = PolyCollection([list(zip(freq, spect))], **kwargs)
        poly._facecolors2d = poly._facecolors
        poly._edgecolors2d = poly._edgecolors
        ax.add_collection3d(poly, zs=z, zdir='y')


def define_line_style(waterfall, kwargs):
    if waterfall:
        s_args = {"alpha": 0.5}
        exp_line = {"hatch": "*"}
    else:
        s_args = {"linewidth": 3, "path_effects": [pe.Stroke(linewidth=4, foreground=[0, 0, 0]), pe.Normal()]}
        exp_line = {"linestyle": "--"}
    s_args.update(kwargs)
    exp_args = {}
    exp_args.update(s_args)
    exp_args.update(exp_line)
    return s_args, exp_args


def scale_spectra(energy_unit, spectra, xshift, yscale, rscale):
    x = spectra.x
    y = spectra.y
    freq = spectra.freq
    spect = spectra.spect
    if energy_unit.lower() == "nm":
        x = np.divide(1239.842 * np.ones(x.shape), x)
        freq = np.divide(1239.842 * np.ones(freq.shape), freq)
    elif energy_unit.lower() == "wn":
        x = np.multiply(8100 * np.ones(x.shape), x)
        freq = np.multiply(8100 * np.ones(freq.shape), freq)
    elif energy_unit.lower() == "eh":
        x = np.multiply(8100 * np.ones(x.shape), x)
        freq = np.divide(freq, 27.2114 * np.ones(freq.shape))

    x += xshift
    freq += xshift
    spect *= yscale
    y *= rscale
    return x, y, freq, spect


def plot_initialization(ax, waterfall, colors, xlim, ylim, zlim, xshift, yscale):
    if colors is None:
        colors = color_cycle
    if xlim is not None:
        xlim_mod = [x + xshift for x in xlim]
        ax.set_xlim(xlim_mod)
    if ylim is not None:
        ylim = [y * yscale for y in ylim]
    if waterfall:
        spect_idx = 0
        color_kw = "facecolors"
        ymin = ylim[0]
        ax.set_yticks([])
        temp = ylim
        ylim = zlim
        zlim = temp
    else:
        ymin = 0
        spect_idx = None
        color_kw = "color"
    if ylim is not None:
        ax.set_ylim(ylim)
    if zlim is not None:
        ax.set_zlim(zlim)
    return spect_idx, colors, color_kw, ymin


def create_legend(ax, legend, waterfall):
    lgd_param = {}
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 5 or waterfall:
            max_label_len = np.max(np.char.str_len(labels))
            if max_label_len > 10:
                ncols = 3
            elif max_label_len > 5:
                ncols = 4
            elif max_label_len > 3:
                ncols = 5
            else:
                ncols = 7
            if waterfall:
                lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=ncols, frameon=False)
            else:
                lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=ncols, frameon=False)
        else:
            lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.03, 0.5), frameon=False)
        lgd_param = {"bbox_extra_artists": (lgd,), "bbox_inches": "tight"}
    return lgd_param


def check_energy_units(energy_unit):
    if energy_unit.lower() not in ["ev", "nm", "wn", "eh"]:
        raise TypeError(
            "Plotting energy by  " + energy_unit + " is not currently supported. Please choose a valid unit of energy.")


def define_axis(ax=None, energy_unit: str = "ev", xlabel: str = "Energy", ylabel: str = "Intensity (Arbitrary Units)",
                zlabel: str = None, title: str = None, waterfall: bool = False):
    check_energy_units(energy_unit)
    if ax is None:
        if waterfall:
            ax = plt.figure().add_subplot(projection='3d')
        else:
            ax = plt.figure().add_subplot()
    if waterfall:
        temp = ylabel
        ylabel = zlabel
        zlabel = temp
    if xlabel is not None:
        xlabel += " (" + energy_unit + ")"
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if zlabel is not None and waterfall:
        ax.set_zlabel(zlabel)
        ax.zaxis.labelpad = 7.5
    if title is not None:
        ax.set_title(title)
    figure = ax.get_figure()
    return figure, ax


def x_mask(x, y, xlim):
    x_mask = np.ma.masked_outside(x, xlim[0], xlim[1]).mask
    filtered_x = x[~x_mask]
    filtered_y = y[~x_mask]
    return filtered_x.flatten(), filtered_y.flatten()


def y_mask(x, y, ylim):
    if np.sign(ylim[0]) == np.sign(ylim[1]):
        y_range = np.ma.masked_greater_equal(y, 0).mask
        if np.sign(ylim[0]) == -1:
            y_mask = np.ma.masked_greater(y, ylim[1]).mask
        else:
            y_mask = np.ma.masked_less(y, ylim[0]).mask
        inadequate_y_mask = np.logical_and(y_range, y_mask)
        filtered_x = x[~inadequate_y_mask]
        filtered_y = y[~inadequate_y_mask]
    else:
        filtered_x = x
        filtered_y = y
    lower_y_mask = np.ma.masked_less(filtered_y, ylim[0])
    filtered_y = lower_y_mask.filled(ylim[0])
    higher_y_mask = np.ma.masked_greater(filtered_y, ylim[1])
    filtered_y = higher_y_mask.filled(ylim[1])
    return filtered_x.flatten(), filtered_y.flatten()


def filter_spect_lim(axis_type, spectra_dict, lim, keep_all):
    filtered_spectra = {}
    if axis_type == "y":
        mask_meth = y_mask
    else:
        mask_meth = x_mask
    for name, spectra in spectra_dict.items():
        if spectra.x is not None:
            filtered_x, filtered_y = mask_meth(spectra.x, spectra.y, lim)
        filtered_freq, filtered_spect = mask_meth(spectra.freq, spectra.spect, lim)
        if filtered_spect.any() or keep_all:
            if isinstance(spectra, sp.ImportedSpectrum):
                filtered_spectra[name] = sp.ImportedSpectrum(filtered_freq, filtered_spect)
            else:
                filtered_spectra[name] = sp.Spectrum(filtered_freq, filtered_spect, filtered_x, filtered_y)
    return filtered_spectra


def find_limit(spectra_dict, axis_type):
    if axis_type:
        value_list = [current_spectra.spect.reshape(-1, 1) for current_spectra in spectra_dict.values()]
    else:
        value_list = [current_spectra.freq.reshape(-1, 1) for current_spectra in spectra_dict.values()]
    value_list = np.vstack(value_list)
    proposed_lim = (np.min(value_list), np.max(value_list))
    return proposed_lim


def find_lim(axis_type, spectra, paired_experimental):
    axis_type = int(axis_type == "y")
    sim_lim = find_limit(spectra, axis_type)
    if paired_experimental:
        experimental_lim = find_limit(paired_experimental, axis_type)
        sim_lim = (np.min([sim_lim, experimental_lim]), np.max([sim_lim, experimental_lim]))
    marg = (sim_lim[1] - sim_lim[0]) * plt.margins()[axis_type]
    lim = (sim_lim[0] - marg, sim_lim[1] + marg)
    return lim


def plot(ax, spectra: dict, paired_experimental: dict = {}, xlim: tuple = None, ylim: tuple = None, zlim: tuple = None, xshift: int = 0,
         yscale: int = 1, rscale: int = 1,  energy_unit: str = "ev", sticks: bool = True, lines: bool = True,
         legend: bool = True, show: bool = True, keep_all=False, save_title: str = None, colors=None,  **kwargs):
    waterfall = ax.name == "3d"
    check_energy_units(energy_unit)
    if xlim is None:
        xlim = find_lim("x", spectra, paired_experimental)
    filtered_spect_dict = filter_spect_lim("x", spectra, xlim, keep_all)
    filtered_experimental_dict = filter_spect_lim("x", paired_experimental, xlim, keep_all)
    if ylim is None:
        ylim = find_lim("y", filtered_spect_dict, filtered_experimental_dict)
    if waterfall:
        spectra = filter_spect_lim("y", filtered_spect_dict, ylim, keep_all)
        paired_experimental = filter_spect_lim("y", filtered_experimental_dict, ylim, keep_all)
    spect_idx, colors, color_kw, ymin = plot_initialization(ax, waterfall, colors, xlim, ylim, zlim, xshift, yscale)
    s_args, exp_args = define_line_style(waterfall, kwargs)

    for current_entry, color in zip(spectra.items(), colors.values()):
        (current_spectrum_name, current_spectrum) = current_entry
        experimental_spectrum = paired_experimental.get(current_spectrum_name)
        color_arg = {color_kw: color}
        s_args.update(color_arg)
        exp_args.update(color_arg)

        if isinstance(current_spectrum, sp.ImportedSpectrum):
            freq = current_spectrum.freq
            spect = current_spectrum.spect
        else:
            x, y, freq, spect = scale_spectra(energy_unit, current_spectrum, xshift, yscale, rscale)
            if sticks:
                plot_sticks(x, y, ax, color, ymin, spect_idx)
        if lines:
            plot_line(ax, freq, spect, spect_idx, label=current_spectrum_name, **s_args)
            if experimental_spectrum is not None:
                if waterfall:
                    spect_idx += 1
                plot_line(ax, experimental_spectrum.freq, experimental_spectrum.spect, spect_idx,
                          label=current_spectrum_name + " (Exp)", **exp_args)
        if waterfall:
            spect_idx += 1
    lgd_param = create_legend(ax, legend, waterfall)
    if save_title is not None:
        plt.savefig(save_title, **lgd_param)
    if show:
        plt.show()


def gen_spect_batch(spectra: dict, broad: float = 0.5, wlim=None, res: float = 100, xshift: float = 0, meth: str = 'lorentz'):
    for current_spectrum in spectra.values():
        if isinstance(current_spectrum, sp.SimulatedSpectrum):
            current_spectrum.gen_spect(broad, wlim, res, xshift, meth)


def gen_spect_batch_mp(spectra: dict, broad: float = 0.5, wlim=None, res: float = 100, xshift: float = 0, meth: str = 'lorentz'):
    simulated_list = [current_spectrum for current_spectrum in spectra.values() if isinstance(current_spectrum, sp.SimulatedSpectrum)]

    with Pool() as pool:
        results = pool.map(gen_spect_wrapper, ((current_simulated, broad, wlim, res, xshift, meth) for current_simulated in simulated_list))
    return results


def gen_spect_wrapper(arg):
    current_spectrum, broad, wlim, res, xshift, meth = arg
    current_spectrum.gen_spect(broad, wlim, res, xshift, meth)
    return current_spectrum

