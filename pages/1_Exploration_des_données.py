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
    - [Source des données](#source)
    - [Répartition des fichiers dans les datasets](#r-partition-des-fichiers-dans-les-datasets)
    - [Répartition des numéros de série ID machine](#r-partition-des-id-machine)
    - [Représentation des sons](#repr-sentation-des-sons)
    - [Spectrogramme MEL](#spectrogrammes-mel)
    ''',
    unsafe_allow_html=True)

@st.cache
def get_data():
    return True

with header:
    st.title('Exploration des données')

    st.header('Source')
    st.write('Kaggle: https://www.kaggle.com/datasets/daisukelab/dc2020task2')

with body:
    st.header('Répartition des fichiers dans les datasets')
    st.image(Image.open('./images/files_distribution.png'), caption ='Répartition des fichier normaux/anormaux')
    st.markdown(
        '''
        Les **datasets d'entrainement** ne contiennent **que des sons normaux**, avec un catégorisant le **numéro de serie de la machine**.<br>
        Les **datasets de test contiennent** des **sons normaux et anormaux**, toujours avec le label du **numéro de serie de la machine**.<br>
        ''',
        unsafe_allow_html=True
    )

    st.header('Répartition des ID machine')
    st.image(Image.open('./images/id_distribution.png'), caption ='Répartition des numéros de série ID machine')
    st.markdown(
        '''
        La **répartition des séries** de machine est **équilibrée** dans les datasets d'entrainement et de test.
        ''',
        unsafe_allow_html=True
    )

    st.header('Représentation des sons')
    col_1, col_2 = st.columns(2)

    col_1.image(Image.open('./images/time_frequency.png'), caption ='Décomposition des dimensions du son')
    col_2.markdown(
        '''
        On distingue **3 principaux types** de représentation d'un son:
        * **temporelle**: variation de l’amplitude au cours du temps
        * **fréquencielle**: décompositions des fréquences en fonction de l'amplitude
        * **en spectrogramme**: permet de représenter les 3 dimensions 
        ''',
        unsafe_allow_html=True
    )

    st.header('Spectrogrammes MEL')
    st.subheader('Le meilleur choix')
    col_1, col_2 = st.columns(2)

    col_1.image(Image.open('./images/spectral_domain_mel_valve.png'), caption ='Spectrogramme MEL du dataset VALVE')
    col_2.image(Image.open('./images/spectral_domain_mel_toyconveyor.png'), caption ='Spectrogramme MEL du dataset TOYCONVEYOR')

    st.markdown(
        '''
        **Meilleures représentations des différents datasets lors de l\'EDA:**
        ''',
        unsafe_allow_html=True
    )
    df = pd.read_csv('./images/eda_summary.csv')
    st.table(df)