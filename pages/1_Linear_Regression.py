# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:48:14 2023

@author: User2021
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
from module_1 import InitializeData, LinearRegression
from module_1 import CV_ArrPreparation, CV_ValidationLoss

st.set_page_config(page_title="Linear Regression", page_icon="🌊")

# Designing content of sidebar
with st.sidebar:
  st.markdown("# Inputting Data")
  data = st.file_uploader("Data File", type=["xlsx", "xls"])
  degree = st.number_input("Polynom degree", min_value=int(1))
  proceed = st.button("Submit data")
  
# Designing main page
if proceed:
  # 1. Training process and showing linear regression model
  
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
  
  # Text
  st.markdown("Regression model result")
  st.text("This is the plot of training data and the linear regression model.")
  # Plotting 
  fig, ax = plt.subplots()
  ax.plot(x_model, t_model)
  ax.set_xlabel('Year')
  ax.set_ylabel('Temperature')
  ax.set_title('Regression Model')
  ax.scatter(x_ori,t_ori, s=3, c="r")
  st.pyplot(fig)
  
  # 2. Cross Validation
  k = 10 # ingat bahwa soal hanya meminta k=10 dan LOOCV dengan k = len(df)
  degree = 11
  x,t = InitializeData(df, degree)
  
  x_training_list, t_training_list, x_validation_list, t_validation_list, w_list = CV_ArrPreparation (k, x, t, degree)
  validation_loss = CV_ValidationLoss (k, x_validation_list, t_validation_list, w_list)
  
  # 3. Prediction
  x_prediksi = st.number_input("Value to predict", min_value=float(0))
