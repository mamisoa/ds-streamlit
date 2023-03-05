from distutils.log import info
import streamlit as st
import os, os.path
import time
import pickle

import pandas as pd
import numpy as np
import tensorflow as tf

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

# seaborn
import seaborn as sns
sns.set_theme()

from PIL import Image


from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, classification_report
from skimage.metrics import structural_similarity as sk_ssim

datasets=['toycar','toyconveyor','fan','pump','slider','valve']
machine_dict = {'fan':[0,2,4,6],'pump':[0,2,4,6],'slider':[0,2,4,6],'toycar':[1,2,3,4],'toyconveyor':[1,2,3],'valve':[0,2,4,6]}


# sidebar
sidebar = st.sidebar
sidebar.subheader('Démonstration')
ds = sidebar.selectbox('Selectionner le dataset:',
    options = datasets
    )

with sidebar.expander("See explanation"):
    st.markdown("""
        1. Predict the machine ID
        2. Dectect anomalous sounds
            1. Group sounds by predicted machine ID
            2. Calculate the threshold of similarity:
                * pass the train set in the autoencoder
                * get the similarity threshold at percentile 90
            3. Pass the sound test files in the corresponding AE ID model
                * mark anomalous files if similarity is less than similarity threshold
        3. Report results for the full test set
    """)

header = st.container()
body = st.container()
footer = st.container()

# charger un modèle
# charger un dataset de test
# calculer les prédictions
# faire un tableau de classification
st.title(f'{ds.capitalize()} dataset analysis')

def predict_id_model(ds,verbose=False):
    '''Machine ID prediction model'''
    st.write('Predicting machine id in dataset: ',ds)
    cnn = tf.keras.models.load_model(f'./models/id_predictions/{ds}/guess_id_{ds}')
    npzfileid = []
    for root,dirs,files in os.walk('./data/'+ds):
        for file in files:
            if file.startswith('test_set_id') and file.endswith('_mfcc.npz'):
                npzfileid.append(os.path.join(root, file))
    for id in npzfileid:
        if verbose == True:
            st.write(f'Loading TEST set {ds}:', id)

    X_test = np.vstack([np.load(file)['arr_0'] for file in npzfileid])
    if verbose == True:
        st.write('X_test  shape:', X_test.shape)
    
    # create categorical stack related to number of machine ID
    shape0 = [ s.shape for s in [np.load(file)['arr_0'] for file in npzfileid]]
    y_test = np.hstack( [ np.ones(shape=(s,)).astype('int8')+id for id,s in zip( range(len(npzfileid)), [ s0[0] for s0 in shape0])  ] )
    if verbose == True:
        st.write('y_test shape:', y_test.shape)
    y_pred = np.argmax(cnn.predict(X_test),axis=1)+1
    y_pred = np.load(f'./models/id_predictions/{ds}/y_pred_{ds}.npy')
    return y_test,y_pred

a,b = predict_id_model(ds)
st.write('Results of ID predictions:')
report_id = classification_report(a,b, output_dict=True)
st.table(report_id)

# X_train dans l'autoencoder
# déterminer le percentile de similarité 90% qui sera le seuil de normalité
# X_test dans l'autoencoder
# déterminer l'anomalie si similarité inféreure au seuil

infoContainer = st.empty()

def ssim(input_img, output_img):
    '''
    similarity function
    '''
    return 1 - tf.reduce_mean(tf.image.ssim(input_img, tf.cast(output_img, tf.float32), max_val=1))

def get_truth_df(ds,id):
    '''
    Returns a dataframe with prediction against test set
        0 if anomalous
    '''
    with infoContainer:
        st.markdown(
            f'''
            **Grouping predicted {ds} #{id}...**
            ''',
            unsafe_allow_html=True
        )
        # get the threshold
        ae_conv = tf.keras.models.load_model(f'./models/anomaly_detection/{ds}/detect_id_{id}_{ds}', custom_objects={"ssim": ssim })
        X_train = np.load(f'./data/{ds}/train_set_id{id}_mel.npz')['arr_0']
        X_train_ae = ae_conv.predict(X_train)
        ssim_lst = []
        for a , b in zip(X_train,X_train_ae):
            ssim_lst.append(sk_ssim(a.astype('float32'),b,multichannel=True))
        ssim_arr = np.array(ssim_lst)
        threshold_ssim = np.percentile(ssim_arr, 90)
        st.write(f'Calculated threshold for normality for {ds} #{id} : {threshold_ssim:.6f}')
        # predict anomaly 
        X_test = np.load(f'./data/{ds}/test_set_id{id}_mel.npz')['arr_0']
        X_test_ae = ae_conv.predict(X_test)
        y_test = np.load(f'./data/{ds}/test_set_id{id}_mel.npz')['arr_1']
        contamination = len(y_test[y_test==0])/y_test.shape[0]
        st.write(f'Numbers of anomalous files in {ds} #{id} : {len(y_test[y_test==0])} out of {y_test.shape[0]} (contamination = {contamination*100:.2f}%)')
        ssim_lst = []
        for a , b in zip(X_test,X_test_ae):
            ssim_lst.append(sk_ssim(a.astype('float32'),b,multichannel=True))
        ssim_arr = np.array(ssim_lst)
        # st.write('ssim_arr shape:',ssim_arr.shape)
        # st.write('y_test shape:', y_test.shape)
    df_ae = pd.DataFrame({
        'predictedNormal': [ 1 if s > threshold_ssim else 0 for s in ssim_arr],
        'isNormal': y_test
        })
    return df_ae

for i,id in enumerate(machine_dict[ds]):
    if i == 0:
        df_ae = get_truth_df(ds,machine_dict[ds][i])
    else:
        df_ae = pd.concat([ df_ae, get_truth_df(ds,machine_dict[ds][i]) ])

infoContainer.write('')

report = classification_report(df_ae['predictedNormal'],df_ae['isNormal'], output_dict=True)

col_1, col_2 = st.columns(2)

with col_1:
    st.metric(label='Samples', value=df_ae.shape[0])
    st.metric(label='Anomalies', value=len(df_ae[df_ae['isNormal']==0]))
    contamination = len(df_ae[df_ae['isNormal']==0])/df_ae.shape[0]
    st.metric(label="Contamination", value=f"{contamination*100:.2f}%")
    # st.write(f"Numbers of anomalous files: {len(df_ae[df_ae['isNormal']==0])} out of {df_ae.shape[0]} (contamination = {contamination*100:.2f}%)")
    

with col_2:
    precision_0 = report['0']['precision']
    recall_0 = report['0']['recall']
    f1_0 = report['0']['f1-score']
    st.metric(label="Precision", value=f"{precision_0*100:.2f}%")
    st.metric(label="Recall", value=f"{recall_0*100:.2f}%")
    st.metric(label="F1", value=f"{f1_0*100:.2f}%")
    
st.table(report)

col_3, col_4 = st.columns(2)

for root,dirs,files in os.walk('./data/'+ds):
        for file in files:
            if file.startswith('normal') and file.endswith('.wav'):
                audio_file = open(os.path.join(root, file), 'rb')
                audio_bytes = audio_file.read()
                col_3.write(file)
                col_3.audio(audio_bytes, format='audio/wav')
            if file.startswith('anomaly') and file.endswith('.wav'):
                audio_file = open(os.path.join(root, file), 'rb')
                audio_bytes = audio_file.read()
                col_4.write(file)
                col_4.audio(audio_bytes, format='audio/wav')