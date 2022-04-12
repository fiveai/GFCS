# (LB plotting/stat code)
import configargparse
from pathlib import Path
from os import listdir
from os.path import isdir, isfile, join, splitext

import torch
import matplotlib.pyplot as plt
import json
from ours_util import stats_summary, bootstrap_sampling, MAX_QUERIES


def process_result_file(result, args):
    input = Path(result)
    input_data = torch.load(input)

    result_path = Path(args.result_dir)

    is_success = input_data['succs']
    queries = input_data['queries']

    # Parse and summarise results for this run
    all_queries, succs_blows, summary = stats_summary(is_success, queries)

    medians_stdev, cdf_perc025, cdf_perc975 = bootstrap_sampling(succs_blows, all_queries)

    summary['queries_median_stdev'] = medians_stdev

    # Save summary as json
    with open(result_path / ('summary_' + splitext(input.name)[0] + '.json'), 'w') as summary_out:
        json.dump(summary, summary_out)

    # Pretty print summary to stdout
    print('######  Summary for run {}'.format(result))
    print('Median query count: {} +/- {:.2}'.format(summary['queries_median'], summary['queries_median_stdev']))
    print('Success rate: {:.2%}'.format(summary['success_rate']))
    print()

    # Empirical PDF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_hist, bins_hist, _ = ax.hist(succs_blows, range=[0, MAX_QUERIES], bins=args.bins, histtype='stepfilled')
    plt.xlabel("Queries per image")
    plt.ylabel("Samples fooled at each query")

    if args.display_plots:
        plt.show()
        plt.close()

    # Empirical CDF
    fig = plt.figure()
    ax = fig.add_subplot(111)
    n_cumhist, bins_cumhist, _ = ax.hist(succs_blows, range=[0, MAX_QUERIES], bins=args.bins, cumulative=True,
                                         histtype='stepfilled')

    plt.xlabel("Queries per image")
    plt.ylabel("Cumulative sum of samples fooled")

    if args.display_plots:
        plt.show()
        plt.close()

    # Save empirical distributions to file for later consumption
    hist_data = {'n_hist': n_hist, 'bins_hist': bins_hist}
    cum_hist_data = {'n_cumhist': n_cumhist, 'bins_cumhist': bins_cumhist, 'perc025_cumhist': cdf_perc025,
                     'perc975_cumhist': cdf_perc975}

    torch.save({'hist_data': hist_data, 'cum_hist_data': cum_hist_data},
               result_path / ('hist_data_' + splitext(input.name)[0] + '.pt'))


def plot_results():

    parser = configargparse.ArgumentParser(
        description="Method that reads in an output structure from a batch of results, processes it, and draws plots.",
        add_help=False
    )

    required_named_arguments = parser.add_argument_group("required named arguments")
    required_named_arguments.add_argument('--input', type=str, required=True,
        help="The name of the result file to be processed and plotted, including any required path info."
    )

    optional_arguments = parser.add_argument_group("optional arguments")

    # We switched help off in order to get our required arguments before our optional ones. Now we just add it back:
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
        '--display_plots', action='store_true',
        help="If specified, display the resulting PDF and CDF. It requires the user to close the window to continue "
             "running. (default: '%(default)s')"
    )
    optional_arguments.add_argument(
        '--bins', type=int, default=250,
        help="The number of bins in the histogram. (default: %(default)d)"
    )
    optional_arguments.add_argument(
        '--result_dir', type=str, default='experimental_results/2022/all_processed_results',
        help="Directory for saving results. (default: '%(default)s')"
    )

    args = parser.parse_args()

    if isfile(args.input):
        process_result_file(args.input, args)
    elif isdir(args.input):
        list_of_results = [join(args.input, r) for r in listdir(args.input) if isfile(join(args.input, r)) and
                           join(args.input, r).endswith('.pt')]
        for result in list_of_results:
            process_result_file(result, args)
    else:
        raise TypeError('--input needs to be a result file or folder of result files.')


if __name__ == "__main__":

    plot_results()
