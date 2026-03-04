import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from sf_analysis import obtain_informative_sf
import warnings
warnings.filterwarnings('ignore')


# Define the semi-factual problem
class SemiFactualProblem(ElementwiseProblem):
    def __init__(self, test_instance, key_feature_index, classifier, label, xl, xu, copula, min_threshold, max_threshold):
        self.test_instance = test_instance
        self.key_feature_index = key_feature_index
        self.classifier = classifier
        self.label = label
        self.copula = copula
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        # normalizing factor for f2
        self.num_other_features = len(self.test_instance) - 1
        super().__init__(n_var=test_instance.shape[0], n_obj=2, n_constr=3, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Objective 1: Maximize the distance on the key feature (it is already normalized between 0-1)
        f1 = abs(x[self.key_feature_index] - self.test_instance[self.key_feature_index])

        # Objective 2: Minimize the distance on other features
        f2 = np.sum([abs(x[i] - self.test_instance[i]) for i in range(len(x)) if i != self.key_feature_index])

        # Constraint: New instance must belong to the same class (here checking with the actual label of the test instance)
        predicted_class = self.classifier.predict(x.reshape(1, -1))[0]
        g1 = int(predicted_class != self.label)

        # Add a new constraint such that the log-likelihood value is less than a threshold (min log-pdf value)
        log_val = self.copula.log_probability_density(x)
        # Ensure log-likelihood is within [threshold_min, threshold_max]
        g2_min = self.min_threshold - log_val  # Ensure log_val >= threshold_min
        g2_max = log_val - self.max_threshold  # Ensure log_val <= threshold_max

        out["F"] = [-f1, f2]
        # Set the constraint, g1 must be 0 for the solution to be feasible
        # (i.e. the generated instance belongs to the same class as test instance)
        out["G"] = [g1, g2_min, g2_max]


# Function to run NSGA-II for each key feature
def run_optimization(test_instance, key_feature_index, classifier, label, xl, xu, copula, min_threshold, max_threshold, explainer, tau_threshold):
    algorithm = NSGA2(
        pop_size=50,
        eliminate_duplicates=True
    )

    # Define the problem for the current feature
    problem = SemiFactualProblem(test_instance, key_feature_index, classifier, label, xl, xu, copula, min_threshold, max_threshold)

    # Run the optimization
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 100),
                   seed=42,
                   verbose=False)

    # Get the Pareto front solutions
    pareto_front = res.X

    ########################### New Approach ################################
    '''
    Select the best solution based on trend strength of key-feature contribution change from query to sf
    (could also use a weighted approach here based on trend strength and minimum objective values) - TO DO
    '''
    if (len(pareto_front) > 0):
        sf_type_list = []
        tau_list = []
        hidden_feature_list = []
        # for each solution (sf) in the pareto front, analyze the SHAP feature contribution trend and identify Boundary-SF
        for sol in pareto_front:
            sf_type, kendall_tau, hidden_feature = obtain_informative_sf(test_instance,
                                                             sol, key_feature_index, label, explainer, classifier,
                                                             tau_threshold)
            sf_type_list.append(sf_type)
            tau_list.append(kendall_tau)
            hidden_feature_list.append(hidden_feature)

        sf_set = set(sf_type_list)

        # for selecting the best solution based on knee-index (minimum objective values)
        # Define the ideal point (can be set as the minimum values in the objective space)
        ideal_point = np.min(res.F, axis=0)
        # Compute Euclidean distance from each point to the ideal point
        distances = np.linalg.norm(res.F - ideal_point, axis=1)
        # Find the index of the solution with the minimum distance
        knee_index = np.argmin(distances)

        ################### In case of all Fail-SF or Not-Informative-SF or mix of both ############################
        if sf_set.issubset({"Fail-SF", "Not-Informative-SF"}):
            best_kendall_tau = None
            support_feature = None
            # Retrieve the knee-point and the corresponding solution
            best_sf = res.X[knee_index]
            best_sf_type = 'Fail-SF/Not-Informative-SF'

        else:
            ############ If Pareto solution contains atleast one Informative-SF type ###############
            # Filter valid indices for Boundary-SF
            valid_indices = [i for i, sf in enumerate(sf_type_list) if
                             sf in {"Informative-SF"}]

            valid_tau_list = [tau_list[i] for i in valid_indices]
            valid_sf_types = [sf_type_list[i] for i in valid_indices]
            # if all tau values are same and they all belong to Boundary-SF type, get the one based on knee-index
            if len(set(valid_tau_list)) == 1 and len(set(valid_sf_types)) == 1:
                # check if knee index in valid indices if not get the first index on valid indices (this is arbitary)
                best_index = knee_index if knee_index in valid_indices else valid_indices[0]
            else:
                # select the index as the best index with the highest absolute trend value (kendall tau)
                best_index = max(valid_indices,
                                 key=lambda i: abs(tau_list[i]))

            best_sf = pareto_front[best_index]
            best_sf_type = sf_type_list[best_index]
            support_feature = hidden_feature_list[best_index]
            best_kendall_tau = tau_list[best_index]

    else:
        best_sf = None
        best_sf_type = None
        support_feature = None
        best_kendall_tau = None

    return key_feature_index, best_sf, best_sf_type, best_kendall_tau, support_feature