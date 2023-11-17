# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:48:14 2023

@author: User2021
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module_1 import InitializeData, LinearRegression, Predict
from module_1 import CV_ArrPreparation, CV_ValidationLoss, AllLossPlot

st.set_page_config(page_title="Linear Regression", page_icon="🌊")

# Designing content of sidebar
with st.sidebar:
  st.markdown("# Inputting Data")
  data = st.file_uploader("Data File", type=["xlsx", "xls"])
  degree = st.number_input("Polynom degree", min_value=int(1))
  
  if "data_submitted" not in st.session_state:
    proceed = st.button("Submit data")
    if proceed:
      st.session_state.data_submitted = True
  else:
    proceed = True
  
# Designing main page
if proceed:
  tab1, tab2, tab3 = st.tabs(["Regression model", "Prediction", "Cross-validation"])
  
  # 1. Training process and showing linear regression model
  with tab1:
    # Import data
    df = pd.read_excel(data)
    
    # Inisialisasi data
    x, t = InitializeData(df, degree)
    
    # Mencari parameter w dengan regrelsi linear
    w = LinearRegression(x, t, degree)
    np_array = df.to_numpy()
    
    # Variabel untuk plot hasil dari model regresi
    x_model, t_model = [], []
    for i in range (len(df)):
      x_model.append(x[i,1])
      
    t_baru_matriks = np.matmul(x,w)
    
    for i in range (len(df)):
      t_model.append(t_baru_matriks[i][0])
    
    # Variabel berisi nilai x dan t pada data original
    x_ori, t_ori= [], []
    
    for i in range (len(df)):
      x_ori.append(np_array[i,0])
      t_ori.append(np_array[i,1])
  
    st.session_state.model_done = True
  
    # Text output
    st.markdown("# Regression model result")
    st.text("This is the plot of training data and the linear regression model.")
    # Plotting 
    fig, ax = plt.subplots()
    ax.plot(x_model, t_model)
    ax.set_xlabel('Year')
    ax.set_ylabel('Temperature')
    ax.set_title('Regression Model')
    ax.scatter(x_ori,t_ori, s=3, c="r")
    st.pyplot(fig)
  
  # 2. Prediction
  with tab2:
    st.markdown("# Predicting response")
    x_prediksi = st.number_input("Predictor value", min_value=float(min(x_ori)))
    t_prediksi = Predict(x_prediksi, degree, w)
    st.text("Predicted value: " + str(t_prediksi))
  
  # 3. Cross Validation
  with tab3:
    st.markdown("# Cross-validation Results")
    st.markdown("## 10-Fold CV")
    
    k = 10
    # Plotting training and validation loss
    degree_list, training_loss_list, validation_loss_list = AllLossPlot (df,k)
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    ax[0].plot(degree_list, training_loss_list)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[1].plot(degree_list, validation_loss_list)
    ax[1].set_title("Validation Loss")
    ax[1].set_xlabel("Degree")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    st.pyplot(fig)
    
    st.markdown("## LOOCV (Leave one out cross-validation)")
    
    k = len(df) 
    # Plotting training and validation loss
    degree_list, training_loss_list, validation_loss_list = AllLossPlot (df,k)
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    ax[0].plot(degree_list, training_loss_list)
    ax[0].set_title("Training Loss")
    ax[0].set_xlabel("Degree")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)
    ax[1].plot(degree_list, validation_loss_list)
    ax[1].set_title("Validation Loss")
    ax[1].set_xlabel("Degree")
    ax[1].set_ylabel("Loss")
    ax[1].grid(True)
    st.pyplot(fig)
