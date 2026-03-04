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



def obtain_results(ensemble_sf_types):
    '''

    Args:
        informative semi-factuals for each baseline semi-factual method

    Returns:
        values for each evaluation metrics


    '''
    new_dict = []
    for item in tqdm(ensemble_sf_types):
        informative_sf_methods = {}
        methods = item['methods']

        # Step 1: Find all methods with sf_type = 'Informative-SF'
        for method_name, method_data in methods.items():
            if method_data.get('sf_type') == 'Informative-SF':
                informative_sf_methods[method_name] = method_data

        # If no Informative-SF methods found, return None
        if not informative_sf_methods:
            new_dict.append({
                'query': item['query'],
                'query_orig': item['query_orig'],
                'label': item['class'],
                'sf_type': None,
                'hidden_feature': None,
                'kendall_tau': None
            })

        else:
            # Step 2: Check if any Informative-SF methods have hidden_feature not None
            methods_with_support = {
                name: data for name, data in informative_sf_methods.items()
                if data.get('hidden_feature') is not None
            }

            if methods_with_support:
                # Select the one with lowest kendall_tau among those with hidden_feature not None
                selected_method_name, selected_method_data = min(
                    methods_with_support.items(),
                    key=lambda x: x[1].get('kendall_tau', float('inf'))
                )
            else:
                # All have hidden_feature = None, select the one with lowest kendall_tau
                selected_method_name, selected_method_data = min(
                    informative_sf_methods.items(),
                    key=lambda x: x[1].get('kendall_tau', float('inf'))
                )

            new_dict.append({
                'query': item['query'],
                'query_orig': item['query_orig'],
                'label': item['class'],
                'method': selected_method_name,
                'sf': selected_method_data['sf'],
                'sf_type': selected_method_data['sf_type'],
                'sf_query': selected_method_data['sf_query'],
                'sparsity': selected_method_data['sparsity'],
                'sparsity_new': selected_method_data['sparsity_new'],
                'ood': selected_method_data['ood'],
                'trust': selected_method_data['trust'],
                'key_feature': selected_method_data['key_feature'],
                'kendall_tau': selected_method_data['kendall_tau'],
                'hidden_feature': selected_method_data['hidden_feature']
            })

    sf_seesaw_pattern(new_dict)
    compute_metrics_values(new_dict)
