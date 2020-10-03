import optuna
import neptune
import neptunecontrib.monitoring.optuna as optuna_utils
from dlrm_s_pytorch_class import DLRM_Model


neptune.init(api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODViMDk1NTktYzM0ZC00MDMwLTgxMzItOTU4YmM3ODkwMjdjIn0=',
             project_qualified_name='pedrobaiz/dlrm')


neptune.create_experiment('optuna-sweep')
neptune_callback = optuna_utils.NeptuneCallback()

def objective(trial):

    # Suggest values of the hyperparameters using a trial object.
    den_fea = 13
    n_bot_layers = trial.suggest_int('n_bot_layers', 2, 5)
    n_top_layers = trial.suggest_int('n_top_layers', 2, 4)
    bot_layers = []
    top_layers = []
    arch_sparse_feature_size = trial.suggest_int('arch_sparse_feature_size', 16, 32)
    for i in range(n_bot_layers):
        if i == 0:
            bot_layers.append(den_fea)  # This value is related to the number of numerical columns (fixed by input data)
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
    #loss_function = trial.suggest_categorical('loss_function', ['mse', 'bce'])
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1)
    print('MLP bot', arch_mlp_bot)
    print('MLP top', arch_mlp_top)
    dlrm_model = DLRM_Model(
        arch_sparse_feature_size=arch_sparse_feature_size, #16,
        arch_mlp_bot=arch_mlp_bot, #'13-512-256-64-16',
        arch_mlp_top=arch_mlp_top, #'512-256-1',
        data_generation='dataset',
        data_set='kaggle',
        raw_data_file='./input/trainday0day0.txt',
        #processed_data_file='./input/kaggleAdDisplayChallenge_processed.npz',
        loss_function='bce', #loss_function,
        round_targets=True,
        learning_rate=learning_rate,
        mini_batch_size=128,
        print_freq=256,
        test_freq=128,
        mlperf_logging=True,
        print_time=True,
        test_mini_batch_size=256,
        test_num_workers=16
        # enable_profiling=True,
        # plot_compute_graph=True,
        # save_model='dlrm_criteo_kaggle.pytorch'
    )
    validation_results = dlrm_model.run()
    for key in validation_results:
        if key not in ['classification_report', 'confusion_matrix']:
            neptune.log_metric(key, validation_results[key])
    return validation_results['best_pre_auc_test'] # ['best_auc_test']  #


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[neptune_callback])
optuna_utils.log_study(study)


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


