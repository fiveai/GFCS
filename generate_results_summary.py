# (LB plotting/stat code)
# This consumes the results output by plot_results so that multiple PDF/CDFs can be plotted together.

import configargparse
from pathlib import Path
import json
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
import torch

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
# for Palatino and other serif fonts use:
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

_TITLE_SIZE = 10
_LEGEND_SIZE = 10
_AXIS_LABEL_SIZE = 10
_TICK_SIZE = 10
_LINE_WIDTH = 0.7
_FIGSIZE = (4, 3)

_LISTS_OF_COLORS = [
    ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive',
     'tab:cyan', 'gold', 'lightcoral', 'black', 'red'],  # 0
    ['tab:blue', 'red', 'black', 'green', 'purple', 'brown', 'orange'],  # 1
    # Seaborn Spectral palette
    ['#f98e52', '#ffffbe', '#86cfa5'],  # 2
    ['#e2514a', '#fca55d', '#fee999', '#edf8a3', '#a2d9a4', '#47a0b3'],  # 3
    ['#dd4a4c', '#f98e52', '#fed481', '#ffffbe', '#d6ee9b', '#86cfa5', '#3d95b8'],  # 4
    # Seaborn Flare palette
    ['#e5715e', '#c14168', '#863071'],  # 5
    ['#e98d6b', '#e3685c', '#d14a61', '#b13c6c', '#8f3371', '#6c2b6d'],  # 6
    ['#ea916e', '#e5715e', '#d9535d', '#c14168', '#a3386f', '#863071', '#672a6b'],  # 7
    # Seaborn Icefire palette
    ['#4167c7', '#1f1e1e', '#b93540'],  # 8
    ['#55a3cd', '#4954b0', '#282739', '#3b2127', '#9c2f45', '#e96f36'],  # 9
    ['#60abcd', '#4167c7', '#37355c', '#1f1e1e', '#5c2935', '#b93540', '#ed7e40'],  # 10
    # Seaborn HSV palette
    ['#ffd500', '#4fff00', '#00ff86', '#009eff', '#3700ff', '#ff00ed'],  # 11
    ['#ffbd00', '#84ff00', '#00ff39', '#00fff6', '#004bff', '#7200ff', '#ff00cf'],  # 12
    # Seaborn default palette (deep)
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],  # 13
    ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],  # 14
    # Seaborn default palette (colorblind)
    ['#0173b2', '#de8f05', '#029e73', '#d55e00', '#cc78bc', '#ca9161'],  # 15
    # Seaborn default palette (dark)
    ['#001c7f', '#b1400d', '#12711c', '#8c0800', '#591e71', '#592f0d', '#a23582']  # 16
]


def _get_yticks(expm_name):

    if 'untargeted' in expm_name:
        if 'incep3' in expm_name:
            return [600, 800, 1000, 1200, 1500, 2000]
        else:
            if 'vgg16' in expm_name and '1surr' in expm_name:
                return [1000, 1200, 1500, 1700, 2000]
            elif 'vgg16' in expm_name and '4surr' in expm_name:
                return [1200, 1500, 1700, 2000]
            elif 'resnet50' in expm_name and '1surr' in expm_name:
                return [1000, 1200, 1500, 1700, 2000]
            elif 'resnet50' in expm_name and '4surr' in expm_name:
                return [800, 1000, 1200, 1500, 1700, 2000]
            elif 'simbaPCA' in expm_name:
                return [200, 500, 1000, 2000]
    elif 'simbaPCA' in expm_name:
        return [200, 500, 1000, 2000]
    elif 'targeted' in expm_name:
        return [1, 10, 100, 1000]
    else:
        return []


def generate_results_summary():

    parser = configargparse.ArgumentParser(
        description="Method that reads multiple outputs of plot_results.py and draw plots of PDFs and CDFs for paper",
        add_help=False
    )

    required_named_arguments = parser.add_argument_group("required named arguments")

    required_named_arguments.add_argument(
        '--expm_json', type=str,
        help="path to json file containing instructions to plot the summarises of multiple experiments together."
    )

    optional_arguments = parser.add_argument_group("optional arguments")
    optional_arguments.add_argument(
        '-h', '--help', action='help', default=configargparse.SUPPRESS,
        help="show this help message and exit"
    )
    optional_arguments.add_argument(
        '--config_file', is_config_file=True,
        help="Optional file from which to read parameter values. In the case of multiple specifications, the override "
             "order is (command line) > (environment vars) > (config file) > (defaults), as in the ConfigArgParse "
             "docs. See the docs for the valid config file format options."
    )
    optional_arguments.add_argument(
        '--limit_nbins', type=int, default=-1,
        help="Limit the displayed xrange to zoom in a section of the histogram (-1 means use all). (default: "
             "%(default)d)"
    )
    optional_arguments.add_argument(
        '--save_to', type=str,
        help="if specified, saves output result to this location"
    )
    optional_arguments.add_argument(
        '--plot_title', type=str,
        help="if specified, give a title to the mosaic of plots"
    )

    optional_arguments.add_argument(
        '--color_palette', type=int, default=0,
        help="Palette used to draw the set of lines forming the CDFs"
    )

    args = parser.parse_args()

    # Read json file with the configurations for plotting multiple experiments together
    with open(Path(args.expm_json)) as json_config:
        all_expm = json.load(json_config)

    plt.rc('axes', prop_cycle=(cycler('color', _LISTS_OF_COLORS[args.color_palette])))

    # Draw all the CDFs together
    fig = plt.figure(figsize=_FIGSIZE)
    ax_cdf_log = fig.add_subplot(111)

    if args.plot_title:
        fig.suptitle(args.plot_title, fontsize=_TITLE_SIZE)
    fig.tight_layout()

    expm_labels = []

    for expm in all_expm:
        color = expm['color'] if 'color' in expm else None
        linestyle = '-'
        expm_labels.append(expm['label'])
        histogram_data = torch.load(expm['hist_output_path'])
        hw_cdf = histogram_data['cum_hist_data']['bins_cumhist'][1] - histogram_data['cum_hist_data']['bins_cumhist'][0]
        limit_cdf = args.limit_nbins if args.limit_nbins != -1 else len(histogram_data['cum_hist_data']['n_cumhist'])

        # work out bin centers
        bin_centers_cumhist = histogram_data['cum_hist_data']['bins_cumhist'] + hw_cdf/2
        this_hist_data = histogram_data['cum_hist_data']['n_cumhist'][:limit_cdf]

        p = ax_cdf_log.plot(
            bin_centers_cumhist[:limit_cdf], this_hist_data,
            linewidth=_LINE_WIDTH,
            color=color,
            ls=linestyle
        )
        color_just_used = p[-1].get_color()
        ax_cdf_log.fill_between(bin_centers_cumhist[:limit_cdf],
                                histogram_data['cum_hist_data']['perc025_cumhist'],
                                histogram_data['cum_hist_data']['perc975_cumhist'],
                                color=color_just_used, alpha=.15, linewidth=0)

    ax_cdf_log.set_xlim([20, 10000])
    ax_cdf_log.tick_params(axis='both', labelsize=_TICK_SIZE)
    ax_cdf_log.set_xlabel("Queries per image", fontsize=_AXIS_LABEL_SIZE)
    ax_cdf_log.set_ylabel("Number fooled", fontsize=_AXIS_LABEL_SIZE)
    ax_cdf_log.set_xscale('log')
    ax_cdf_log.set_yscale('log')
    custom_yticks = _get_yticks(args.expm_json)
    if custom_yticks:
        ax_cdf_log.set_yticks(custom_yticks)

    ax_cdf_log.set_xticks([20, 50, 100, 1000, 10000])
    ax_cdf_log.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_cdf_log.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax_cdf_log.minorticks_off()
    ax_cdf_log.legend(expm_labels, fontsize=_LEGEND_SIZE)

    if args.save_to:
        fig.savefig(args.save_to, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    generate_results_summary()
