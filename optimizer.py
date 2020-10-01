import optuna
import neptune
import neptunecontrib.monitoring.optuna as optuna_utils

from dlrm_s_pytorch_class import DLRM_Model


neptune.init(api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODViMDk1NTktYzM0ZC00MDMwLTgxMzItOTU4YmM3ODkwMjdjIn0=',
             project_qualified_name='pedrobaiz/dlrm')



dlrm_model = DLRM_Model()


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


