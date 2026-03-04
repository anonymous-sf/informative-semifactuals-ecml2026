import os
import pickle
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from utils import process_item
from tqdm import tqdm
from analyze_results import obtain_results


if __name__ == '__main__':

    data = 'diabetes'
    df = pd.read_csv('../../data/'+data+'_sc.csv')

    target = 'Outcome'
    y = np.array(df[target])
    X = np.array(df.loc[:, df.columns != target])

    # train RF classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X, y)

    # load best query-sf pairs
    file = data+'.pickle'
    with open(file, 'rb') as f:
        results = pickle.load(f)


    # Initialize SHAP Explainer
    explainer = shap.TreeExplainer(classifier)
    tau_threshold = 0.3

    num_workers = 5
    # Using ProcessPoolExecutor to process each item in parallel
    print('Processing ...')
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_item, item, explainer, classifier, tau_threshold) for item in results]

    # Collect the results from each feature optimization
    updated_results = []
    for future in concurrent.futures.as_completed(futures):
        item = future.result()
        updated_results.append(item)

    # obtain results
    obtain_results(updated_results)


