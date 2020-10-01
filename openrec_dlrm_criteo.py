from tensorflow.data import Dataset
from openrec.tf2.recommenders import DLRM
from tensorflow.keras import optimizers
from tqdm import tqdm
import tensorflow as tf
import openrec_dataloader
import os

# pip install neptune-client neptune-contrib['monitoring']
import optuna
import neptune
import neptunecontrib.monitoring.optuna as optuna_utils

neptune.init(api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiODViMDk1NTktYzM0ZC00MDMwLTgxMzItOTU4YmM3ODkwMjdjIn0=',
             project_qualified_name='pedrobaiz/dlrm')

#raw_data = openrec_dataloader.load_criteo('../dataset/')
main_dir = os.getcwd()  # os.path.dirname(os.getcwd()))
raw_data = openrec_dataloader.load_criteo(os.path.join(main_dir, 'input'))
dim_embed = 16 #4
bottom_mlp_size = [13, 512, 256, 64, 16] #[8, 4]
top_mlp_size = [512, 256, 1] #[128, 64, 1]
total_iter = int(1e5)
batch_size = 256 #1024
eval_interval = 100
save_interval = eval_interval

# Sample 1000 batches for training
train_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_train'][:batch_size*1000],
                    'sparse_features': raw_data['X_cat_train'][:batch_size*1000],
                    'label': raw_data['y_train'][:batch_size*1000]
                }).batch(batch_size).prefetch(1).shuffle(5*batch_size)
    
# Sample 100 batches for validation
val_dataset = Dataset.from_tensor_slices({
                    'dense_features': raw_data['X_int_val'][:batch_size*100],
                    'sparse_features': raw_data['X_cat_val'][:batch_size*100],
                    'label': raw_data['y_val'][:batch_size*100]
             }).batch(batch_size)

optimizer = optimizers.Adam()

dlrm_model = DLRM(
                m_spa=dim_embed,
                ln_emb=raw_data['counts'],
                ln_bot=bottom_mlp_size,
                ln_top=top_mlp_size,
                # Other Options
                loss_func = 'bce', #'mse'
                arch_interaction_op = 'dot',
                arch_interaction_itself = False,
                sigmoid_bot = False,
                sigmoid_top = True,
                loss_threshold = 0.0
                # Original
                #arch_interaction_op = None,
                #arch_interaction_itself = False,
                #sigmoid_bot = -1,
                #sigmoid_top = -1,
                #loss_threshold = 0.0,
             )

auc = tf.keras.metrics.AUC()

@tf.function
def train_step(dense_features, sparse_features, label):
    with tf.GradientTape() as tape:
        loss_value = dlrm_model(dense_features, sparse_features, label)
    gradients = tape.gradient(loss_value, dlrm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dlrm_model.trainable_variables))
    return loss_value

@tf.function
def eval_step(dense_features, sparse_features, label):
    pred = dlrm_model.inference(dense_features, sparse_features)
    auc.update_state(y_true=label, y_pred=pred)

average_loss = tf.keras.metrics.Mean()

for train_iter, batch_data in enumerate(train_dataset):
    
    loss = train_step(**batch_data)
    average_loss.update_state(loss)
    print('%d iter training.' % train_iter, end='\r')
    
    if train_iter % eval_interval == 0:
        for eval_batch_data in tqdm(val_dataset,
                                    leave=False, 
                                    desc='%d iter evaluation' % train_iter):
            eval_step(**eval_batch_data)
        print("Iter: %d, Loss: %.2f, AUC: %.4f" % (train_iter, 
                                                   average_loss.result().numpy(),
                                                   auc.result().numpy()))
        average_loss.reset_states()
        auc.reset_states()


#######################################################################
#
# Although it sounds very good we will not follow this because it is an implementation of DLRM in TF
# Unfortunately, its results are not as good. Using very similar parameters to Pytorch, results are:
# Iter: 0, Loss: 0.69, AUC: 0.5812
# Iter: 100, Loss: 0.53, AUC: 0.7015
# Iter: 200, Loss: 0.51, AUC: 0.7088
# Iter: 300, Loss: 0.51, AUC: 0.7123
# Iter: 400, Loss: 0.51, AUC: 0.7117
# Iter: 500, Loss: 0.52, AUC: 0.7129
# Iter: 600, Loss: 0.51, AUC: 0.7133
# Iter: 700, Loss: 0.52, AUC: 0.7145
# Iter: 800, Loss: 0.51, AUC: 0.7144
# Iter: 900, Loss: 0.51, AUC: 0.7145
# This is already improved values from the original ones which give:
# Iter: 0, Loss: 0.22, AUC: 0.5299
# Iter: 100, Loss: 0.19, AUC: 0.6465
# Iter: 200, Loss: 0.18, AUC: 0.6817
# Iter: 300, Loss: 0.17, AUC: 0.6956
# Iter: 400, Loss: 0.17, AUC: 0.7007
# Iter: 500, Loss: 0.17, AUC: 0.7039
# Iter: 600, Loss: 0.17, AUC: 0.7059
# Iter: 700, Loss: 0.17, AUC: 0.7073
# *** The results using the same dataset for Original DLRM is 0.7709 ***
# *** which clearly demonstrates a problem with this approach        ***
#######################################################################