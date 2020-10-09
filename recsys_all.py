from recsys_optimizer import RecsysOptimeyez
from dlrm_s_pytorch_class import DLRM_Model


# Optimise Hyperparameter with Optuna/Neptune with Smaller DataSet
def find_best_parameters():
    params = {
        "verbose": True,
        "max_evals": 10,
        "API_KEY": 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODViMDk1NTktYzM0ZC00MDMwLTgxMzItOTU4YmM3ODkwMjdjIn0=',
        "den_fea": 13
    }
    opt = RecsysOptimeyez(**params)
    return opt.find_best_params()


# Train with Larger DataSet (using hyperparameters from previous step)
def train_recsys(best_hyper_params):

    return 0


# Evaluate Recommendations for Sample Data
def evaluate_recsys(model):

    return 0


if __name__ == "__main__":
    best_hyper_params = find_best_parameters()
    recsys_model = train_recsys(best_hyper_params)
    print('All Good')