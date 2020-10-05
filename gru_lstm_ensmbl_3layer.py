#!/usr/bin/env python
# coding: utf-8

# This notebook shows you how to build a model for predicting degradation at various locations along RNA sequence. 
# * We will first pre-process and tokenize the sequence, secondary structure and loop type. 
# * Then, we will use all the information to train a model on degradations recorded by the researchers from OpenVaccine. 
# * Finally, we run our model on the public test set (shorter sequences) and the private test set (longer sequences), and submit the predictions.
# 
# For more details, please see [this discussion post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/182303).
# 
# ---
# 
# Updates:
# 
# * V7: Updated kernel initializer, embedding dimension, and added spatial dropout. Changed epochs and validation split. All based on [Tucker's excellent kernel](https://www.kaggle.com/tuckerarrants/openvaccine-gru-lstm#Training) (go give them an upvote!).
# * V8-9: Changed loss from MSE to M-MCRMSE, which is the official competition metric. See [this post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183211) for more details. Changed number of epochs to 100.
# * V10: loss function: M-MCRMSE -> MCRMSE
# * V11: Filter `signal_to_noise` to be greater than 1, since the same was applied to private set. See [this post](https://www.kaggle.com/c/stanford-covid-vaccine/discussion/183992). 
# * V12: Loss function: MCRMSE -> M-MCRMSE.
# * V13: Decrease and stratify validation set based on `SN_filter`. Increase embedding size, GRU dimensions, number of layers. All inspired from Tucker's work again.

# In[2]:


import json

import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
# from keras.layer import LSTM, Bidirectional

import tensorflow as tf
# from tensorflow import keras
from sklearn.model_selection import train_test_split

import tensorflow.keras.layers as L
import keras.backend as K

# ## Set seed to ensure reproducibility

# In[2]:


tf.random.set_seed(2020)
np.random.seed(2020)


# ## Helper functions and useful variables


# In[3]:


# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']


# In[4]:


y_true = tf.random.normal((32, 68, 3))
y_pred = tf.random.normal((32, 68, 3))


# In[5]:


# def MCRMSE(y_true, y_pred):
#     colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)
#     return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)
def rmse(y_actual, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_actual, y_pred)
    return K.sqrt(mse)

def mcrmse(y_actual, y_pred, num_scored=len(pred_cols)):
    score = 0
    for i in range(num_scored):
        score += rmse(y_actual[:, :, i], y_pred[:, :, i]) / num_scored
    return score
# In[6]:


def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(L.LSTM(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer = 'orthogonal'))

def build_model_ensmbl(seq_len=107, pred_len=68, dropout=0.5, embed_dim=100, hidden_dim=256, type=0):
    inputs = L.Input(shape=(seq_len, 3))
    
    # split categorical and numerical features and concatenate them later.
    categorical_feat_dim = 3
    categorical_fea = inputs[:, :, :categorical_feat_dim]
    numerical_fea = inputs[:, :, 3:]

    embed = L.Embedding(input_dim=len(token2int), output_dim=embed_dim)(categorical_fea)
    reshaped = tf.reshape(embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))
    reshaped = L.concatenate([reshaped, numerical_fea], axis=2)
    
    if type == 0:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 1:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 2:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 3:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif type == 4:
        hidden = gru_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif type == 5:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    elif type == 6:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = gru_layer(hidden_dim, dropout)(hidden)
    elif type == 7:
        hidden = lstm_layer(hidden_dim, dropout)(reshaped)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
        hidden = lstm_layer(hidden_dim, dropout)(hidden)
    
    truncated = hidden[:, :pred_len]
    out = L.Dense(5, activation='linear')(truncated)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(tf.keras.optimizers.Adam(), loss=mcrmse)
    return model

# In[8]:


def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


# In[9]:


def preprocess_inputs(df, token2int, cols=['sequence', 'structure', 'predicted_loop_type']): #,'bpps_sum','bpps_max','bpps_nb']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in seq])
    )


# ## Load and preprocess data

# In[10]:


data_dir = ''
train = pd.read_json('train.json', lines=True)
test = pd.read_json('test.json', lines=True)
sample_df = pd.read_csv('sample_submission.csv')


# In[11]:

train = train.query("signal_to_noise >= 1")

# In[12]:


# We will use this dictionary to map each character to an integer
# so that it can be used as an input in keras
token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_inputs = preprocess_inputs(train, token2int)
train_labels = pandas_list_to_array(train[pred_cols])

# In[]: additional features

def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"bpps/{mol_id}.npy").max(axis=1))
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(f"bpps/{mol_id}.npy").sum(axis=1))
    return bpps_arr

def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(f"bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    return bpps_arr 

# train_inputs = pd.DataFrame(train_inputs)
# train_inputs['bpps_sum'] = read_bpps_sum(train)
# train_inputs['bpps_max'] = read_bpps_max(train)
# train_inputs['bpps_nb'] = read_bpps_nb(train)
# train_inputs = train_inputs.values



# In[13]:

x_train, x_val, y_train, y_val = train_test_split(
    train_inputs, train_labels, test_size=.1, random_state=34, stratify=train.SN_filter)


# Public and private sets have different sequence lengths, so we will preprocess them separately and load models of different tensor shapes.

# In[14]:

public_df = test.query("seq_length == 107")
private_df = test.query("seq_length == 130")

public_inputs = preprocess_inputs(public_df, token2int)
private_inputs = preprocess_inputs(private_df, token2int)

# public_inputs = pd.DataFrame(public_inputs)
# public_inputs['bpps_sum'] = read_bpps_sum(public_df)
# public_inputs['bpps_max'] = read_bpps_max(public_df)
# public_inputs['bpps_nb'] = read_bpps_nb(public_df)
# public_inputs=public_inputs.values

# private_inputs = pd.DataFrame(private_inputs)
# private_inputs['bpps_sum'] = read_bpps_sum(private_df)
# private_inputs['bpps_max'] = read_bpps_max(private_df)
# private_inputs['bpps_nb'] = read_bpps_nb(private_df)
# private_inputs=private_inputs.values

# ## Build and train model
# 
# We will train a bi-directional GRU model. It has three layer and has dropout. To learn more about RNNs, LSTM and GRU, please see [this blog post](https://colah.github.io/posts/2015-08-Understanding-LSTMs/).

# In[15]:
type_len = 1
public_preds_l = []
private_preds_l = []
for type_num in range(type_len):
    model = build_model_ensmbl(type=type_num)
    model.summary()
    
    
    # In[16]:
    
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        batch_size=128,
        epochs=50,
        verbose=10,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(patience=5),
            tf.keras.callbacks.ModelCheckpoint('model_3layer_'+str(type_num)+'.h5')
        ]
    )
    
    
    # ## Evaluate training history
    # 
    # Let's use Plotly to quickly visualize the training and validation loss throughout the epochs.
    
    # In[17]:
    
    fig = px.line(
        history.history, y=['loss', 'val_loss'],
        labels={'index': 'epoch', 'value': 'MCRMSE'}, 
        title='Training History')
    fig.show()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    # ## Load models and make predictions
    
    # Public and private sets have different sequence lengths, so we will preprocess them separately and load models of different tensor shapes. This is possible because RNN models can accept sequences of varying lengths as inputs.
    
    # In[18]:
    
    # Caveat: The prediction format requires the output to be the same length as the input,
    # although it's not the case for the training data.
    model_public = build_model_ensmbl(seq_len=107, pred_len=107, type=type_num)
    model_private = build_model_ensmbl(seq_len=130, pred_len=130,type=type_num)
    
    model_public.load_weights('model_3layer_'+str(type_num)+'.h5')
    model_private.load_weights('model_3layer_'+str(type_num)+'.h5')
    
    
    # In[19]:
    
    public_preds = model_public.predict(public_inputs)
    private_preds = model_private.predict(private_inputs)
    
    public_preds_l.append(public_preds)
    private_preds_l.append(private_preds)
    # ## Post-processing and submit
    # For each sample, we take the predicted tensors of shape (107, 5) or (130, 5), and convert them to the long format (i.e. $629 \times 107, 5$ or $3005 \times 130, 5$):
    
# In[20]:

public_preds_ens=0
private_preds_ens=0
for type_num in range(type_len):
    public_preds_ens +=  public_preds_l[type_num]
    private_preds_ens += private_preds_l[type_num]

public_preds_ens    = public_preds_ens /type_len
private_preds_ens   = private_preds_ens /type_len

preds_ls = []

for df, preds in [(public_df, public_preds_ens), (private_df, private_preds_ens)]:
    for i, uid in enumerate(df.id):
        single_pred = preds[i]

        single_df = pd.DataFrame(single_pred, columns=pred_cols)
        single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

        preds_ls.append(single_df)

preds_df = pd.concat(preds_ls)
preds_df.head()


# In[21]:

submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
submission.to_csv('submission_3layer_ens.csv', index=False)


# In[21]:
for type_num in range(type_len):
    preds_ls = []
    
    for df, preds in [(public_df, public_preds_l[type_num]), (private_df, private_preds_l[type_num])]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
    
            single_df = pd.DataFrame(single_pred, columns=pred_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
    
            preds_ls.append(single_df)
    
    preds_df = pd.concat(preds_ls)
    preds_df.head()
    
    submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])
    submission.to_csv('submission_3layer_'+str(type_num)+'.csv', index=False)