import optuna
import neptune
import warnings
import sys
import neptunecontrib.monitoring.optuna as optuna_utils
from dlrm_s_pytorch_class import DLRM_Model


class RecsysOptimeyez():

    def __init__(self,
                 verbose=True,
                 to_path = "save",
                 bucketname='local', #BUCKET_NAME_LOCAL,
                 max_evals=100,
                 neptune_tags=['local'],
                 run_tag = None,
                 API_KEY = None, #NEPTUNE_API_KEY,
                 model_name = 'model_all',
                 trial_size=20,
                 den_fea = None
                 ):
        self.verbose = verbose
        self.to_path = to_path
        self.bucketname = bucketname
        self.max_evals = max_evals
        self.neptune_tags = neptune_tags
        self.run_tag = run_tag
        self.API_KEY = API_KEY
        self.model_name = model_name
        self.trial_size = trial_size
        if den_fea:
            self.den_fea = den_fea
        else:
            sys.exit(
                "ERROR: argument 'den_fea' should be specified"
            )

    def get_params(self, deep=True):
        return {'verbose': self.verbose,
                'to_path': self.to_path,
                'bucketname': self.bucketname,
                'max_evals': self.max_evals,
                'neptune_tags': self.neptune_tags,
                'run_tag': self.run_tag,
                'API_KEY': self.API_KEY,
                'model_name': self.model_name,
                'trial_size': self.trial_size,
                'den_fea': self.den_fea
                }

    def set_params(self, **params):
        self.__fitOK = False
        for k, v in params.items():
            if k not in self.get_params():
                warnings.warn("Invalid parameter a for data Data(). "
                              "Parameter IGNORED. Check the list of available "
                              "parameters with `data.get_params().keys()`")
            else:
                setattr(self, k, v)

    def rebuild_mlps(self, params, best_params):

        bot_layers = []
        top_layers = []
        for i in range(best_params['n_bot_layers']):
            if i == 0:
                bot_layers.append(self.den_fea)  # This value is related to the number of numerical columns (fixed by input data)
            elif i == (best_params['n_bot_layers'] - 1):
                bot_layers.append(best_params['arch_sparse_feature_size'])  # This value is related to the arch_sparse_feature_size
            else:
                bot_layers.append(best_params['n_bot_units_l{}'.format(i)])
        for i in range(best_params['n_top_layers']):
            if i == (best_params['n_top_layers']-1):
                top_layers.append(1)  # This value should always be 1, as it is a binary classification
            else:
                top_layers.append(best_params['n_top_units_l{}'.format(i)])

        params["arch_sparse_feature_size"] = best_params["arch_sparse_feature_size"]
        params["arch_mlp_bot"] = '-'.join(str(x) for x in bot_layers)
        params["arch_mlp_top"] = '-'.join(str(x) for x in top_layers)
        params["learning_rate"] = best_params["learning_rate"]

        return params

    ##############################################################################
    # AUTOMATED HYPER-TUNING & Model Training
    ##############################################################################
    def find_best_params(self):

        def objective(trial, params):

            # Suggest values of the hyperparameters using a trial object.
            n_bot_layers = trial.suggest_int('n_bot_layers', 2, 5)
            n_top_layers = trial.suggest_int('n_top_layers', 2, 4)
            bot_layers = []
            top_layers = []
            arch_sparse_feature_size = trial.suggest_int('arch_sparse_feature_size', 16, 32)
            for i in range(n_bot_layers):
                if i == 0:
                    bot_layers.append(self.den_fea)  # This value is related to the number of numerical columns (fixed by input data)
                elif i == (n_bot_layers-1):
                    bot_layers.append(arch_sparse_feature_size)  # This value is related to the arch_sparse_feature_size
                else:
                    bot_features = trial.suggest_int('n_bot_units_l{}'.format(i), 32, 512)
                    bot_layers.append(bot_features)
            for i in range(n_top_layers):
                if i == (n_top_layers-1):
                    top_layers.append(1)  # This value should always be 1, as it is a binary classification
                else:
                    top_features = trial.suggest_int('n_top_units_l{}'.format(i), 32, 512)
                    top_layers.append(top_features)
            arch_mlp_bot = '-'.join(str(x) for x in bot_layers)
            arch_mlp_top = '-'.join(str(x) for x in top_layers)
            learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
            #loss_function = trial.suggest_categorical('loss_function', ['mse', 'bce'])

            # Assigning trial hyper-parameters to params
            params["arch_sparse_feature_size"] = arch_sparse_feature_size
            params["arch_mlp_bot"] = arch_mlp_bot
            params["arch_mlp_top"] = arch_mlp_top
            params["learning_rate"] = learning_rate

            # Run DLRM and get results
            dlrm_model = DLRM_Model(**params)
            validation_results = dlrm_model.run()
            for key in validation_results:
                if key not in ['classification_report', 'confusion_matrix']:
                    neptune.log_metric(key, validation_results[key])

            # Print trial (if verbose)
            if self.verbose:
                print('Parameters: ', params, '/n Results: ', validation_results)

            return validation_results['best_pre_auc_test'] # ['best_auc_test'] Need to decide which metric is best

        # Assigning fixed parameters to params
        params = {
            "data_generation": 'dataset',
            "data_set": 'kaggle',
            "raw_data_file": './input/trainday0day0day0day0.txt',
            # "processed_data_file": './input/kaggleAdDisplayChallenge_processed.npz',
            "loss_function": 'bce',  # loss_function,
            #"round_targets": True,  We want to have a ranked list instead of yes/no
            "mini_batch_size": 32,  # 128,
            "print_freq": 32,  # 256,
            "test_freq": 32,  # 128,
            "mlperf_logging": True,
            "print_time": True,
            "test_mini_batch_size": 32,  # 256,
            # "test_num_workers": 16
            # "save_model ":  'dlrm_criteo_kaggle_.pytorch'
            # "use_gpu": True
            # "enable_profiling": True,
            # "plot_compute_graph": True,
        }
        neptune.init('pedrobaiz/dlrm', api_token=self.API_KEY)
        neptune.create_experiment('recsys-' + self.model_name, tags=[str(self.neptune_tags)])
        neptune_callback = optuna_utils.NeptuneCallback()
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, params), n_trials=self.max_evals, callbacks=[neptune_callback])
        optuna_utils.log_study(study)
        best_params = self.rebuild_mlps(params, study.best_params)

        return best_params


# See sample available options for optimising hyper-parameters

##################################################################################
#
# Following this example: https://optuna.org/
#
##################################################################################
# # 1. Define an objective function to be maximized.
# def objective(trial):
#
#     # 2. Suggest values of the hyperparameters using a trial object.
#     n_layers = trial.suggest_int('n_layers', 1, 3)
#     layers = []
#
#     in_features = 28 * 28
#     for i in range(n_layers):
#         out_features = trial.suggest_int('n_units_l{}'.format(i), 4, 128)
#         layers.append(torch.nn.Linear(in_features, out_features))
#         layers.append(torch.nn.ReLU())
#         in_features = out_features
#     layers.append(torch.nn.Linear(in_features, 10))
#     layers.append(torch.nn.LogSoftmax(dim=1))
#     model = torch.nn.Sequential(*layers).to(torch.device('cpu'))
#     ...
#     return accuracy
#
# # 3. Create a study object and optimize the objective function.
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)


##################################################################################
#
# Following this https://docs.neptune.ai/integrations/optuna.html
#
##################################################################################
# neptune.create_experiment('optuna-sweep')
# neptune_callback = optuna_utils.NeptuneCallback()
#
# def objective(trial):
#     data, target = load_breast_cancer(return_X_y=True)
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.25)
#     dtrain = lgb.Dataset(train_x, label=train_y)
#
#     param = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#     }
#
#     gbm = lgb.train(param, dtrain)
#     preds = gbm.predict(test_x)
#     accuracy = roc_auc_score(test_y, preds)
#     return accuracy
#
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
# optuna_utils.log_study(study)


