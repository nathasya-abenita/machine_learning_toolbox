# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 12:04:09 2023

@author: User2021
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module_2 import Euclidean, Manhattan, Hamming
from module_2 import CV_MinimizingLoss, PredictNN, ConfusionMatrix

st.set_page_config(page_title="K-NN", page_icon="🌊")

# Designing content of sidebar
with st.sidebar:
  st.markdown("# Inputting Data")
  data = st.file_uploader("Data File", type=["csv"])
  distance_opt = st.selectbox(
    'Distance Method',
    ('Euclidean', 'Manhattan'))
  
  if "data_submitted_3" not in st.session_state:
    proceed = st.button("Submit data")
    if proceed:
      st.session_state.data_submitted_3 = True
  else:
    proceed = True
    
# Designing main page
if proceed:
  with st.spinner("In progress..."):
    # Calling chosen distance function
    if distance_opt == "Euclidean":
      DistanceFunc = Euclidean
    elif distance_opt == "Manhattan":
      DistanceFunc =  Manhattan
    else:
      DistanceFunc = Hamming
    
    # Reading file
    df = pd.read_csv(data)
    df = df[:150] # mengambil 150 data pertama saja
    np_array = df.to_numpy()
    
    # Mempersiapkan variabel pecahan data
    x_data = np_array[:, 0:len(np_array[0])-2]
    t_data = np.zeros((len(np_array),1))
    for i in range (len(np_array)):
      t_data[i][0] = np_array[i][len(np_array[0])-1]
    
    # Nilai k maksimum yang ingin dicek
    k_max = 20
    
    # Mencari nilai K yang meminimalisasi CV loss
    k_nn_optimal, loss_list = CV_MinimizingLoss (k_max, x_data, t_data, DistanceFunc)
    k_is_found = True
    
    # Output printing
    fig, ax = plt.subplots()
    ax.set_xlabel(r'$k$')
    ax.set_ylabel('loss')
    ax.set_title('CV Loss')
    ax.plot([i for i in range (1,k_max+1)], loss_list)
    ax.grid()
    
    # Predicting classes for all data
    t_new_list = []
    # Mencari predicted class untuk tiap data yang ada
    for id in range (len(x_data)):
      x = x_data[id]
      # Hasil klasifikasi
      t_new = PredictNN(k_nn_optimal, x, x_data, t_data, DistanceFunc)
      t_new_list.append(t_new)
      
    # Finding Confusion Matrix
    # Mencari jumlah kelas
    c = len(df.iloc[:,-1].value_counts().index.tolist())
    conf_matrix = ConfusionMatrix (c, t_data.flatten(), t_new_list)
    
  
  st.markdown("# Performing CV to Find Optimal K")
  st.pyplot(fig)
  st.text("Optimal K: " + str(k_nn_optimal))
  st.markdown("# Confusion Matrix")
  st.write(conf_matrix)

