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

## Useful functions
def show_matrix(y_test,y_pred, model_file, dataset):
  report = classification_report(y_test,y_pred, output_dict=True)
  f1_1, f1_2, f1_3 = report['1']['f1-score'],report['2']['f1-score'],report['3']['f1-score']
  matrix = confusion_matrix(y_test,y_pred)
  ax = plt.axes()
  ax.set_title(f'Model: {model_file.split("_")[2]} Dataset: {dataset} \nF1 "1"={f1_1*100:.2f}% "2"={f1_2*100:.2f}% "3"={f1_3*100:.2f}% ')
  sns.heatmap(pd.DataFrame(matrix),
                xticklabels=range(1,matrix.shape[0]+1), 
                yticklabels=range(1,matrix.shape[1]+1), 
                annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
  plt.ylabel('Predicted')
  plt.xlabel('Actual');

datasets=['fan','pump','slider','toycar','toyconvor','valve']

header = st.container()
body = st.container()
footer = st.container()

# sidebar
sidebar = st.sidebar
sidebar.header('Plan')


sidebar.subheader('Démonstration')
sidebar.selectbox('Selectionner le dataset:',
    options = datasets
    )


@st.cache
def get_data():
    return True



with header:
    st.title('Détection de sons anormaux produits par des machines industrielles')

with body:
    st.header('Dataset')
    st.text('Origin of dataset lorem ipsum dolor sit amet, consectetur adip')
    st.image(Image.open('./images/intro_img.png'), caption ='But du projet')

with footer:
    st.header('Features')
    st.text('Features of dataset lorem ipsum dolor sit amet, consectetur adip')
    st.header('Model training')
    st.text('Train dataset lorem ipsum dolor sit amet, consectetur adip')
    
    sel_col, displ_col = st.columns(2)

    max_depth = sel_col.slider('slider', min_value=20, max_value=100, value=20, step=10)
    
    dataset = sel_col.selectbox('Choose database:',
        options=['fan','pump','slider','toycar','toyconvor','valve'])
    
    input_feat = sel_col.text_input('Write something','Something')
    button1 = st.button('Click me')
    check1 = st.checkbox('Check me')
    submit_button = st.button('Submit')

    displ_col.subheader('Training results:')
    displ_col.write(dataset)
    if button1:
        st.write('Button1 clicked')
    if submit_button:
        if check1:
            st.write('Check is checked')
        else:
            st.write('Check is not checked')