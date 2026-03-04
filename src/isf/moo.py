import os
import concurrent.futures
from moo_optimization import run_optimization

# get final Informative-SF of the test instance based on the filtering evaluation
def get_informative_sf(test, classifier, label, min_vals, max_vals, cat_idx, cat_embed, copula, min_threshold, max_threshold, explainer, tau_threshold):

    # Set the number of workers equal to the number of features
    num_workers = 5

    # run optimization considering each feature as key-feature
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(run_optimization, test, key_feature_index, classifier, label, min_vals, max_vals, copula, min_threshold, max_threshold, explainer, tau_threshold)
            for key_feature_index in range(test.shape[0])
        ]

        # Collect the results from each feature optimization
        results = []
        for future in concurrent.futures.as_completed(futures):
            key_feature_index, best_sf, best_sf_type, best_kendall_tau, support_feature = future.result()  # Unpack all three values
            results.append((key_feature_index, best_sf, best_sf_type, best_kendall_tau, support_feature))

    # Sort the results based on feature index
    results_sorted = sorted(results, key=lambda x: x[0])

    # Extract feature-sfs and their values as list
    feat_idxs = [item[0] for item in results_sorted]
    feat_sfs = [item[1] for item in results_sorted]
    feat_sf_type = [item[2] for item in results_sorted]
    feat_sf_tau = [item[3] for item in results_sorted]
    feat_sf_support = [item[4] for item in results_sorted]

    # transform categorical embedding if exists
    if (len(cat_idx) > 0):
        for item in feat_sfs:
            if item is not None:
                # get the closest categorical embedded value
                for i in cat_idx:
                    val = item[i]
                    cat_list = list(cat_embed[i].values())
                    # get closest
                    diff = lambda list_value: abs(list_value - val)
                    closest = min(cat_list, key=diff)
                    # replace the categorical with the closest
                    item[i] = closest


    return feat_idxs, feat_sfs, feat_sf_type, feat_sf_tau, feat_sf_support


