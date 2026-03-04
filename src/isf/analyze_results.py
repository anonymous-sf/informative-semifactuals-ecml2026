import numpy as np
from tqdm import tqdm
from statistics import mean

def sf_seesaw_pattern(sf_dict):
    '''

    Args:
        sf_dict: dictionary with best ensemble informative-sfs

    Returns: sfs with seesaw pattern

    '''
    total = 0
    count = 0

    for item in sf_dict:
        if item['sf_type'] == 'Informative-SF':
            total = total + 1
            if item['hidden_feature'] is not None:
                count = count + 1

    coverage = (count / total) * 100
    print('Percentage of SF with seesaw patterns: ',coverage)
    print('-----------------')


def min_max_normalize(values):
    """
    Method to Normalize list to [0, 1] range
    """
    if not values:
        return []

    min_val = min(values)
    max_val = max(values)

    normalized = [(x - min_val) / (max_val - min_val) for x in values]
    return normalized

def compute_metrics_values(sf_dict):
    '''

    Args:
        sf_dict: dictionary with best ensemble informative-sfs

    Returns: mean scores for each evaluation metrics

    '''

    sf_query = []
    sparsity = []
    ood = []
    trust = []


    for item in sf_dict:
        if item['sf_type'] == 'Informative-SF' and item['hidden_feature'] is not None:
            sf_query.append(item['sf_query'])
            sparsity.append(item['sparsity_new'])
            ood.append(item['ood'])
            if (item['trust'] < 10):
                trust.append(item['trust'])

    print('Distance: ', mean(sf_query))
    print('-------------------')
    print('Sparsity: ', mean(sparsity))
    print('-------------------')
    print('Plausibility: ', mean(ood))
    print('-------------------')
    print('Trustworthiness: ', mean(min_max_normalize(trust)))



def obtain_results(results):
    '''

    Args:
        informative semi-factuals for each baseline semi-factual method

    Returns:
        values for each evaluation metrics


    '''
    new_dict = []

    for item in tqdm(results):
        feature_sfs = item['feature_sfs']
        cols = len(item['query'])

        # Step 1: Filter only Boundary-SF types
        boundary_sfs = [sf for sf in feature_sfs if sf.get('sf_type') == 'Informative-SF']

        if not boundary_sfs:
            new_dict.append({
                'query': item['query'],
                'label': item['class'],
                'sf': None,
                'sf_type': None,
                'hidden_feature': None,
                'kendall_tau': None
            })
        else:

            # Step 2: Check if any have hidden feature not None
            sfs_with_support = [sf for sf in boundary_sfs if sf.get('hidden_feature') is not None]

            if sfs_with_support:
                # Select the one with lowest kendall_tau among those with support_boundary_idx
                selected_sf = min(sfs_with_support, key=lambda x: x.get('kendall_tau', float('inf')))
            else:
                # All have support_boundary_idx = None, select the one with lowest kendall_tau
                selected_sf = min(boundary_sfs, key=lambda x: x.get('kendall_tau', float('inf')))

            # new sparsity based on threshold
            # Count matching non-key features (up to 2 decimal places) -> could also implement threshold
            tolerance = 1e-2
            num_non_key_features = sum(
                np.abs(selected_sf['sf'][i] - item['query'][i]) <= tolerance
                for i in range(len(item['query'])) if i != selected_sf['key_feature']
            )

            new_dict.append({
                'query': item['query'],
                'label': item['class'],
                'key_feature': selected_sf['key_feature'],
                'sf': selected_sf['sf'],
                'sf_type': selected_sf['sf_type'],
                'kendall_tau': selected_sf['kendall_tau'],
                'hidden_feature': selected_sf['hidden_feature'],
                'sparsity': cols - num_non_key_features,
                'sparsity_new': (1 / (cols - num_non_key_features)),
                'sf_query': selected_sf['sf_query'],
                'ood': selected_sf['ood'],
                'trust': selected_sf['trust']
            })

    sf_seesaw_pattern(new_dict)
    compute_metrics_values(new_dict)
