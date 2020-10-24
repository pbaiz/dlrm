from recsys_optimizer import RecsysOptimeyez
from dlrm_s_pytorch_class import DLRM_Model
import json


RUN_TAG = 'Local_Date'


# Optimise Hyperparameter with Optuna/Neptune with Smaller DataSet
def find_best_parameters():
    params = {
        "verbose": True,
        "max_evals": 3,
        "API_KEY": 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODViMDk1NTktYzM0ZC00MDMwLTgxMzItOTU4YmM3ODkwMjdjIn0=',
        "den_fea": 13
    }
    opt = RecsysOptimeyez(**params)
    return opt.find_best_params()


# Train with Larger DataSet (using hyperparameters from previous step)
def train_recsys(best_hyper_params):
    # Run DLRM and get results with Larger Dataset and Best Parameters
    params = best_hyper_params
    params['save_model'] = RUN_TAG + '_dlrm_fazwaz.pytorch'     # This saves model in dlrm/pytorch
    params['save_model_onnx'] = RUN_TAG + '_dlrm_fazwaz2.pytorch'   # This saves model in onnx
    dlrm_model = DLRM_Model(**params)
    final_results = dlrm_model.run()
    return final_results


# Evaluate Recommendations for Sample Data
def evaluate_recsys(best_hyper_params):
    params = best_hyper_params
    params['load_model'] = RUN_TAG + '_dlrm_fazwaz.pytorch'
    params['load_model_onnx'] = RUN_TAG + '_dlrm_fazwaz2.pytorch'   # This saves model in onnx
    params['inference_only'] = True
    dlrm_model = DLRM_Model(**params)
    check = dlrm_model.infer("input_data")
    # dlrm_model.dlrm.eval()
    return check


if __name__ == "__main__":
    # Find best hyper parameters (using small dataset-
    best_hyper_params = find_best_parameters()
    with open('best_hyper_params.json', 'w') as f:
        json.dump(best_hyper_params, f)
    # Run larger training with best hyperparams
    #final_results = train_recsys(best_hyper_params)
    # Load and Run Model for any input
    #with open('best_hyper_params.json', 'r') as f:
    #    best_hyper_params = json.load(f)
    #check = evaluate_recsys(best_hyper_params)
    print('All Good')