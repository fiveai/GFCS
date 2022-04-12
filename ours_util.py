# (LB plotting/stat code)

import numpy as np
import matplotlib.pyplot as plt

# Using a reserved value to flag a failed query
MAX_QUERIES = 10000
_SUCCS = 0
_FAIL = 1


def stats_summary(is_success, queries):
    summary = {}
    num_attacks = len(is_success)
    # write the query count to a list
    all_query_count = [queries[i].item() for i in range(num_attacks)]
    # before computing the median, set the failed queries to Inf
    all_query_count_for_perc = [all_query_count[i] if is_success[i] else float('inf') for i in range(num_attacks)]
    summary['queries_median'] = np.median(all_query_count_for_perc)
    # Flag as failures: queries already flagged as failed *or* that require more than MAX_QUERIES
    succs_blows = [all_query_count[i] for i in range(num_attacks) if (is_success[i] and all_query_count[i] <= MAX_QUERIES)]
    summary['success_rate'] = len(succs_blows) / num_attacks
    summary['all_query_count'] = all_query_count_for_perc
    return all_query_count_for_perc, succs_blows, summary


def bootstrap_sampling(successful_queries, all_queries, num_simulations=1000):
    y_full, medians = [], []
    for i in range(num_simulations):
        resampled_succ_queries = np.random.choice(successful_queries, replace=True, size=len(successful_queries))
        resampled_queries = np.random.choice(all_queries, replace=True, size=len(successful_queries))
        medians.append(np.median(resampled_queries))
        y, _, _ = plt.hist(resampled_succ_queries, range=[0, MAX_QUERIES], bins=250, cumulative=True, histtype='stepfilled')
        if len(y_full)>0:
            y_full = np.vstack((y_full, y))
        else:
            y_full = y

    medians_stdev = np.std(medians)
    cdf_perc025 = np.percentile(y_full, 2.5, axis=0)
    cdf_perc975 = np.percentile(y_full, 97.5, axis=0)

    return medians_stdev, cdf_perc025, cdf_perc975




