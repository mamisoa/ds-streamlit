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


from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, classification_report, make_scorer, mean_squared_error

datasets=['fan','pump','slider','toycar','toyconvor','valve']

header = st.container()
body = st.container()
footer = st.container()

# sidebar
sidebar = st.sidebar
st.sidebar.markdown(
    '''
    # Sections
    - [Conception](#conception)
    - [Algorithmes d'apprentissage conventionnel](#algorithmes-d-apprentissage-conventionnel)
        - [IF-PCA](#isolation-forest)
    - [Algorithmes d'apprentissage profond](#algorithmes-d-apprentissage-profond)
        - [Autoencodeur](#autoencodeur)
        - [AE-ANN](#autoencodeurs-avec-couches-denses)
        - [AE-CNN](#autoencodeurs-avec-couches-de-convolution)
    
    ''',
    unsafe_allow_html=True)

@st.cache
def get_data():
    return True

with header:
    st.title('Modélisation')

with body:
    st.header('Conception')
    st.markdown(
        '''
        La distribution des datasets en **set d'entrainement** et **set de test**, tous labélisés par un numéros de série ID machine, permet de:
        * utiliser un **modèle de classification supervisé** pour **DETERMINER LE MACHINE ID** à partir du son
        * utiliser un **modèle de classification NON supervisé** pour **DETERMINER SI UN SON EST NORMAL ou NON**
        Ces deux modèles intégré dans une **pipeline** permettront de detecter un son anormal plus finement à partir de **modèles spécifiques du machine ID**.
        ''',
        unsafe_allow_html=True
    )
    st.image(Image.open('./images/pipeline.png'), caption ='Pipeline')

    st.header('Algorithmes d\'apprentissage conventionnel')
    
    st.subheader('Isolation forest')
    col_1, col_2 = st.columns(2)

    col_1.image(Image.open('./images/pca_outliers.png'), caption ='Elimination des outliers par PCA')
    col_1.image(Image.open('./images/pca_reduction.png'), caption ='Réduction des dimensions par PCA')
    col_2.markdown(
        '''
        Pipeline en 3 étapes:
        1. **élimination des outliers par Principal Component Analysis(PCA)**
        2. **réduction des dimensions par PCA**: recherche du seuil expliquant 90\% de la variance
        3. **isolation forest**
            * hyperparamètres par défaut
            * optimisation des hyperparamètres par GridSearch

        ''',
        unsafe_allow_html=True
    )

    st.write('Resultats sur le dataset Valve:')
    col_3, col_4,col_5 = st.columns(3)
    col_3.metric(label="Précision", value="33%")
    col_4.metric(label="Rappel", value="46%")
    col_5.metric(label="F1", value="54%")

    df = pd.read_csv('./images/if_pca_results.csv')
    st.table(df)
    
    st.header('Algorithmes d\'apprentissage profond')

    st.subheader('Autoencodeur')
    st.markdown(
        '''
        Pour créer un modèle non supervisé, nous avons utilisé une architecture de type AE:
        1. **Encodage de l’entrée**: l’encodeur génère une image réduite de l’image d’entrée
        2. **Décodage de l’image**: le décodeur reconstitue une image à partir de l’image réduite générée par l’encodeur
        Un son sera classifié comme normal si l’image décodée est similaire à l’image d’entrée.
        Un son sera donc classifié **anormal si le métrique de similarité est inférieure au seuil choisi**.
        Ce seuil est choisi arbitrairement comme **le percentile 90 de similarité déterminé avec les données d'entraînement.**
        ''',
        unsafe_allow_html=True
    )

    st.subheader('Autoencodeurs avec couches denses')
    col_6, col_7 = st.columns(2)
    col_6.markdown(
        '''
        **Modèle:** AE avec un réseau de neurones artificiels (ANN)
        Architecture:
        * **entrée:** MEL spectrogram en couleur (224,224,3)
        * **encodeur:** une couche dense de 1024 neurones avec une activation RELU
        * **décodeur:** une couche dense de 224*224*3 avec une activation RELU suivie d’une factorisation (reshape) (224,224,3)
        * **sortie:** image (224,224,3)
        **Fonction de perte:** Mean Square Error (MSE)

        ''',
        unsafe_allow_html=True
    )
    col_7.image(Image.open('./images/ae-ann.png'), caption ='Modèle AE-ANN')

    col_8, col_9 = st.columns(2)
    col_8.image(Image.open('./images/ae-ann-model.png'), caption ='Evolution du modèle')
    col_9.markdown(
        '''
        Le modèle AE-ANN est **assez simple** et même s'il permet d'obtenir  des modèles
        avec une **accuracy de 0.85** sur la plupart des datasets, il ne s'améliore pas avec les itérations.
        ''',
        unsafe_allow_html=True
    )

    st.subheader('Autoencodeurs avec couches de convolution')
    col_10, col_11 = st.columns(2)
    col_10.markdown(
        '''
        **Modèle:** AE avec un réseau de neurones artificiels avec convolutions (CNN)
        **Architecture symétrique encodeur/décodeur:**
        * **entrée:** MEL spectrogram en couleur (224,224,3)
        * **encodeur: **
            * 3 couches de convolutions de filtres  de 32,16,8,4 (RELU)
            * suivi chaque fois par une couche de MAX POOLING (réduction)
        * **décodeur:**
            * 3 couches de convolutions de filtres  de 4,8,16,32 (RELU)
            * suivi chaque fois par une couche d’ UPSAMPLING (augmentation)
        * **une couche de convolution** (SIGMOID) avec factorisation pour récupérer un format (224,224,3)
        * **sortie:** image (224,224,3)
        **Fonction de perte:** Structural Similarity Index (SSIM)


        ''',
        unsafe_allow_html=True
    )
    col_11.image(Image.open('./images/ae_cnn.png'), caption ='Modèle AE-CNN')
    col_11.image(Image.open('./images/tc_ae-cnn_model.png'), caption ='Graphiques AE-CNN')
    # col_11.image(Image.open('./images/ae-cnn_results.png'), caption ='Résultats AE-CNN')
    df = pd.read_csv('./images/ae-cnn_results.csv')
    col_11.table(df)
