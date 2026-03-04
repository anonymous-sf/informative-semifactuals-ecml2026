import numpy as np
import pymannkendall as mk

def interpolate_all_features(query, sf, n_steps=10):
    """
    Interpolate all features from the query to the semi-factual (SF).
    If feature-values are same, they remain same
    If feature-values are different, they are interpolated accordingly

    Args:
    query: np.array, the query instance
    sf: np.array, the semi-factual instance
    n_steps: int, number of interpolation steps (default is 10)

    Returns:
    interpolations: np.array, an array of interpolated instances
    """
    interpolations = [query]

    for i in range(1, n_steps + 1):
        interpolated_instance = query * (1 - i / n_steps) + sf * (i / n_steps)
        interpolations.append(interpolated_instance)

    return np.array(interpolations)


# function to obtain Informative-SF
def obtain_informative_sf(query, sf, key_feature, label, explainer, classifier, tau_threshold):
    '''
    The function obtains Informative-SF for a given query which includes identifying 
    It uses Criteria-1 (only using Key-Feature Trend) to do the classification
    '''

    # interpolate features from query to sf
    interpolations = interpolate_all_features(query, sf)

    # get shap values of key-feature for interpolations
    key_f_shap = []
    preds = []

    # get non-key features
    non_key_f = [feature for feature in range(len(query)) if feature != key_feature]
    # dictionary to store shap interaction values for non-key features
    non_key_f_shap = {feature: [] for feature in range(len(query)) if feature != key_feature}

    for item in interpolations:
        # using direct shap values
        # shap_values = explainer(item)
        # key_f_shap.append(shap_values.values[:, label][key_feature])

        # using shap interaction values to get the main effect of the key-feature
        shap_interaction_values = explainer.shap_interaction_values(item)
        key_f_shap.append(shap_interaction_values[key_feature][key_feature][int(label)])

        # get shap interaction values for non-key features
        for idx in non_key_f:
            non_key_f_shap[idx].append(shap_interaction_values[idx][idx][int(label)])

        # get prediction of the interpolated item
        preds.append(classifier.predict(item.reshape(1, -1))[0])

    # get Mann-Kendall results for non-key features
    mk_result_non_key = {feature: [] for feature in range(len(query)) if feature != key_feature}
    for idx in non_key_f:
        mk_result_non_key[idx].append(mk.original_test(non_key_f_shap[idx]))

    kendall_tau = None
    hidden_feature = None
    hidden_feature_value = float('-inf')

    # filter for class-consistency (the predictions of all interpolated items should match the orginial label)
    if all(p == label for p in preds):
        # get result from Mann-Kendall test
        mk_result = mk.original_test(key_f_shap)

        # classify SF based on criteria-1
        if (mk_result.trend == 'decreasing' and
                mk_result.p < 0.05 and
                mk_result.Tau < -tau_threshold):

            sf_type = 'Informative-SF'
            kendall_tau = mk_result.Tau

            # get supporting explanation for Boundary-SF
            for key, value_list in mk_result_non_key.items():
                for item in value_list:
                    if item.trend == 'increasing' and item.p < 0.05:
                        if item.Tau > hidden_feature_value:
                            hidden_feature_value = item.Tau
                            hidden_feature = key

        else:
            # cannot classify as Informative-SF
            sf_type = 'Not-Informative-SF'
            kendall_tau = mk_result.Tau
    else:
        # pediction change during interpolation
        sf_type = 'Fail-SF'

    # could return other stats as well
    return sf_type, kendall_tau, hidden_feature

# Define your helper function
def process_item(item, explainer, classifier, tau_threshold):
    query = item['query']
    cols = len(query)
    label = item['class']
    # loop through each method
    for method, method_data in item['methods'].items():
        # filter None sfs
        if method_data['sf'] is not None and method_data['sf_query'] is not None:
            sf = method_data['sf']
            distances = np.abs(sf - query)
            # Find the key feature (feature with max distance)
            key_feature = np.argmax(distances)
            key_feature_distance = distances[key_feature]

            # Count matching non-key features (up to 2 decimal places) -> could also implement threshold
            tolerance = 1e-2
            num_non_key_features = sum(
                np.abs(sf[i] - query[i]) <= tolerance
                for i in range(len(sf)) if i != key_feature
            )

            # Obtain informative-sf
            sf_type, kendall_tau, hidden_feature = obtain_informative_sf(query, sf, key_feature, label, explainer, classifier,
                                                             tau_threshold)

            method_data['key_feature'] = key_feature
            method_data['key_feature_distance'] = key_feature_distance
            method_data['sparsity'] = cols - num_non_key_features
            method_data['sparsity_new'] = (1 / (cols - num_non_key_features))
            method_data['sf_type'] = sf_type
            method_data['kendall_tau'] = kendall_tau
            method_data['hidden_feature'] = hidden_feature
        else:
            method_data['key_feature'] = None
            method_data['key_feature_distance'] = None
            method_data['sparsity'] = None
            method_data['sf_type'] = None
            method_data['kendall_tau'] = None
            method_data['hidden_feature'] = None

    return item