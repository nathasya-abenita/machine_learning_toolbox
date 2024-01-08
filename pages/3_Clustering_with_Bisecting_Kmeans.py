# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:52:04 2024

@author: User2021
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from module_3 import Normalize, Euclidean, Manhattan, BisectingKMeans, SSE_Clustering

st.set_page_config(page_title="Clustering with Bisecting K-Means", page_icon="🌊")

# Designing content of sidebar
with st.sidebar:
  st.markdown("# Inputting Data")
  data = st.file_uploader("Data File", type=["csv"])
  distance_opt = st.selectbox(
    'Distance Method',
    ('Euclidean', 'Manhattan'))
  # INPUT BISECTING K-MEANS PARAMETERS
  k = st.number_input("Number of clusters", min_value=int(1))
  num_iteration = st.number_input("Number of iterations", min_value=int(1))
  
  if "data_submitted_3" not in st.session_state:
    proceed = st.button("Submit data")
    if proceed:
      st.session_state.data_submitted_3 = True
  else:
    proceed = True
    
# Designing main page
if proceed:
  tab1, tab2 = st.tabs(["Clustering Data", "Demonstration"])
  with tab1:
    df = pd.read_csv(data)
    n_var = len(df.columns) # number of variables
    np_array = df.to_numpy()
    
    # NORMALIZING DATA
    np_array = Normalize (np_array, df)
    
    # CHOOSING DISTANCE
    if distance_opt == "Euclidean":
      DistanceFunc = Euclidean
    elif distance_opt == "Manhattan":
      DistanceFunc =  Manhattan
    
    # PERFORMING BISECTING K-MEANS
    cluster_list, centroid_list = BisectingKMeans (k, np_array, num_iteration, DistanceFunc, n_var)
    
    # OUTPUT
    st.markdown("# Clustering with Bisecting K-Means")
    st.text("Dimension of data: " + str(n_var))
    
    # Mengeluarkan anggota tiap cluster
    for i in range (len(cluster_list)):
      st.text("Cluster " + str(i+1) +":")
      for j in range (len(cluster_list[i])):
        st.text(cluster_list[i][j])
      st.text("")
    
    # Mengeluarkan nilai SSE
    SSE = SSE_Clustering(cluster_list, centroid_list, DistanceFunc)[1]
    st.text("")
    st.text("SSE: " + str(SSE))
    
  with tab2:
    st.text("Given data is extracted to be two-dimensional.")
    fig = plt.figure(figsize=[5,5])
    ax = fig.add_subplot(projection='3d')
    
    for cluster_id in range (len(cluster_list)):
      cluster = cluster_list[cluster_id]
      x = []
      y = []
      z = []
      for data in cluster:
        x.append(data[0])
        y.append(data[1])
        z.append(data[2])
      ax.scatter(x, y, z, marker='o', label='cluster ' + str(cluster_id))
    
    ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    ax.set_title("Clustering Result")
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1])
    ax.set_zlabel(df.columns[2])
    st.pyplot(fig)