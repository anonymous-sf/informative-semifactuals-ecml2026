from sklearn.neighbors import NearestNeighbors
from metrics import *
from utils import load_data
from tqdm import tqdm
import pickle
import shap
from moo import get_informative_sf
from sklearn.ensemble import RandomForestClassifier
from copulas.multivariate import GaussianMultivariate
from analyze_results import obtain_results
import warnings
warnings.filterwarnings('ignore')
from pymoo.config import Config
Config.warnings['not_compiled'] = False


def sf_loop(X, y, classifier, nbrs, trust_model, cat_idx, cat_embed, model_path, min_vals, max_vals, min_threshold, max_threshold, explainer, tau_threshold):

    # load copula model
    copula = GaussianMultivariate.load(model_path)

    # create dictionary to hold results
    results = []

    # for all instances in the dataset
    for id, (x_item, y_item) in tqdm(enumerate(zip(X, y)), total=len(X)):
        query = x_item
        # get the prediction of the test instance
        # label = classifier.predict(query.reshape(1, -1))[0]
        # using the actual label
        label = y_item

        # get final mdn for the test instance
        feat_idxs, feat_sfs, feat_sf_type, feat_sf_tau, feat_sf_support = get_informative_sf(query, classifier, label, min_vals, max_vals, cat_idx, cat_embed, copula, min_threshold, max_threshold, explainer, tau_threshold)

        feature_sfs = []
        for idx in range(len(feat_sfs)):
            key_feature = feat_idxs[idx]
            sf = feat_sfs[idx]
            sf_type = feat_sf_type[idx]
            kendall_tau = feat_sf_tau[idx]
            hidden_feature = feat_sf_support[idx]

            if sf is not None:
                # evaluation metrics
                # Sparsity
                sparsity = calculate_sparsity(sf, query)
                # SF-Query Distance
                sf_query = calculate_l2_dist(sf, query)
                # OOD Distance
                ood_dist, _ = calculate_ood(nbrs, y, label, sf)
                # print(ood_dist)
                # Trust score
                trust = trust_model.get_score(np.reshape(sf, (1, -1)), np.reshape(label, (1, -1)))
                # print(trust[0][0])
            else:
                sparsity = None
                sf_query = None
                trust = None
                ood_dist = None

            feature_sfs.append({
                'key_feature': key_feature,
                'sf': sf,
                'sf_type': sf_type,
                'kendall_tau': kendall_tau,
                'hidden_feature': hidden_feature,
                'sparsity': sparsity,
                'sf_query': sf_query,
                'ood': ood_dist,
                'trust': trust[0][0]
            })

        results.append({
            'idx': id,
            'query': query,
            'class': label,
            'feature_sfs': feature_sfs
        })

    return results



if __name__ == '__main__':

    # Specifications
    data_desc = 'german_credit'
    target = 'class'
    method = 'informative_sf'

    X, y, min_vals, max_vals, df, cols = load_data(data_desc, target)

    cat_idx = [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19]
    with open('german_cat_embed.pickle', 'rb') as f:
        cat_embed = pickle.load(f)

    # Fit Gaussian Copula
    copula = GaussianMultivariate()
    copula.fit(df)
    # save copula model
    model_path = data_desc+'_copula.pkl'
    copula.save(model_path)

    # Set adaptive threshold range for plausibility based on mean and std of log-pdf of dataset
    mean_log = np.mean(copula.log_probability_density(X))
    std_log = np.std(copula.log_probability_density(X))
    # Define a scaling factor for delta (adjust based on desired strictness)
    delta_factor = 1.5
    adaptive_delta = delta_factor * std_log
    # Adaptive threshold range
    min_threshold = mean_log - adaptive_delta
    max_threshold = mean_log + adaptive_delta

    # train a RF classifier
    classifier = RandomForestClassifier()
    classifier.fit(X, y)

    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(classifier)
    tau_threshold = 0.3

    # Fit the Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=len(X)).fit(X)

    # initialization for trust score
    trust_model = TrustScore()
    trust_model.fit(X, y)

    results = sf_loop(X, y, classifier, nbrs, trust_model, cat_idx, cat_embed, model_path, min_vals, max_vals, min_threshold, max_threshold, explainer, tau_threshold)

    # obtain results
    obtain_results(results)