# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:52:43 2024

@author: User2021
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Function to normalize data

def Normalize (np_array, df):
  #Mengambil elemen maksimum dan minimum pada setiap kolom menjadi sebuah array
  min_array = df.min().values
  max_array = df.max().values

  #Normalisasi
  for j in range (len(np_array[0])):
    for i in range (len(np_array)):
      np_array[i][j] = (np_array[i][j] - min_array[j]) / (max_array[j] - min_array[j])

  return(np_array)

# Distance

def Euclidean (x, y):
  sum = 0
  if len(x) == len(y):
    for i in range (len(x)):
      sum += (x[i] - y[i])**2
    sum = sum**(1/2)
  else:
    print(len(x))
    print(len(y))
    sum = "Error"
  return sum

def Manhattan (x, y):
  sum = 0
  if len(x) == len(y):
    for i in range (len(x)):
      sum += np.absolute((x[i] - y[i]))
  else:
    print(len(x))
    print(len(y))
    sum = "Error"
  return sum

# SSE, Centroid, and Bisecting K-Means
# Menghitung SSE untuk satu cluster saja
def SSE_Cluster (cluster, centroid, DistanceFunc):
  sse_cluster = 0
  for data in cluster:
    sse_cluster += DistanceFunc(centroid, data)**2
  return sse_cluster

# Menghitung SSE untuk tiap cluster pada cluster_list
def SSE_Clustering(cluster_list, centroid_list, DistanceFunc):
  sse_cluster_list = []
  for i in range (len(cluster_list)):
    cluster = cluster_list[i]
    centroid = centroid_list[i]
    sse_cluster = SSE_Cluster(cluster, centroid, DistanceFunc)
    sse_cluster_list.append(sse_cluster)
  return [sse_cluster, sum(sse_cluster_list)] #[list of SSE in each clusters, SSE as the sum of all SSE]

# Mencari centroid untuk tiap cluster pada cluster_list
def FindCentroid (cluster_list, n_var):
  centroid_list = []
  for cluster in cluster_list:
    centroid = [] # array yang menyimpan koordinat centroid
    for i in range (n_var):
      sum = 0
      for data in cluster:
        sum += data[i]
      mean = sum/len(cluster) # n_var adalah input awal yang dianggap sebagai global variable
      # (tidak perlu jadi input fungsi untuk menghemat waktu)
      centroid.append(mean)
    centroid_list.append(centroid)
  return centroid_list

# Melakukan K-means dengan K=2 untuk suatu cluster yang diberikan
def KMeans (cluster, DistanceFunc, n_var):
  centroid_list = []

  # Inisialisasi dengan memastikan tidak ada cluster kosong (kalo sistem ini mau dihapus, tinggal tulis isEmpty = True di akhir blok)
  isEmpty = False
  for _ in range (2):
    centroid = []
    for _ in range (n_var):
      centroid.append(random.uniform(0,1))
    centroid_list.append(centroid)
  cluster_list = [[], []]
  for data in cluster:
    if DistanceFunc(data, centroid_list[0]) < DistanceFunc(data, centroid_list[1]):
      cluster_list[0].append(data)
    else:
      cluster_list[1].append(data)
  if len(cluster_list[0]) == 0 or len(cluster_list[1]) == 0:
    isEmpty = True

  if not isEmpty:
    # Melakukan K-means hingga kesetimbangan tercapai atau terbentuk cluster kosong
    isFixed = False # menyatakan kesetimbangan tercapai atau belum
    isEmpty = False
    while isFixed == False and isEmpty == False:
      centroid_list = FindCentroid(cluster_list, n_var)
      cluster_list_new = [[], []]
      error = 0
      for data in cluster:
        if DistanceFunc(data, centroid_list[0]) < DistanceFunc(data, centroid_list[1]):
          cluster_list_new[0].append(data)
        else:
          cluster_list_new[1].append(data)
      # Computing error
      centroid_list_new = FindCentroid(cluster_list, n_var)
      for centroid_id in range (2): # hanya ada 2 centroid karena ada 2 cluster saja
        for i in range (n_var):
          error += abs(centroid_list_new[centroid_id][i] - centroid_list[centroid_id][i])
      # Updating new clusters
      cluster_list = cluster_list_new
      # Checking stability
      if error == 0:
        isFixed = True
      # Checking empty cluster
      if len(cluster_list[0]) == 0 or len(cluster_list[1]) == 0:
        isEmpty = True

  # Special treatment for empty cluster (memilih titik dengan SSE tertinggi sebagai centroid baru)
  else:
    # Finding new centroid
    sse_points = []
    for cluster in cluster_list:
      if len(cluster) > 0:
        centroid = FindCentroid([cluster], n_var)[0]
        #print("centroid lama", centroid)
        for data in cluster:
          sse_points.append(DistanceFunc(data, centroid))
        centroid_new_id = np.argmax(sse_points)
        centroid_new = cluster[centroid_new_id]
        #print("centroid baru", centroid_new)
        centroid_list = [centroid, centroid_new]

        # Reassigning the cluster again
        cluster_list = [[], []]
        for data in cluster:
          if DistanceFunc(data, centroid_list[0]) < DistanceFunc(data, centroid_list[1]):
            cluster_list[0].append(data)
          else:
            cluster_list[1].append(data)
  return cluster_list

# Melakukan bisecting K-means dengan memanggil fungsi-fungsi sebelumnya
def BisectingKMeans (k, data_normalized_list, num_iteration, DistanceFunc, n_var):
  cluster_list = [data_normalized_list]
  centroid_list = FindCentroid(cluster_list, n_var)
  for i in range (1,k):
    #print(i, cluster_list)
    # choosing a cluster based on the largest SSE
    SSE_list = SSE_Clustering(cluster_list, centroid_list, DistanceFunc)[0]
    cluster_id = np.argmax(SSE_list)
    cluster = cluster_list[cluster_id]
    centroid = centroid_list[cluster_id]
    # deleting taken cluster and centroid from the list
    cluster_list.pop(cluster_id)
    centroid_list.pop(cluster_id)
    # collecting several alternatives of bisecting cluster
    cluster_new_list = []; SSE_cluster_new_list = []
    for _ in range (num_iteration):
      clusters_new = KMeans(cluster, DistanceFunc, n_var)
      cluster_new_list.append(clusters_new)
      SSE_cluster_new_list.append(SSE_Clustering(clusters_new, FindCentroid(clusters_new, n_var), DistanceFunc)[0])
    # choosing cluster with lowest SSE

    #SSE_cluster_new_list = SSE_Clustering(cluster_list, centroid_list, DistanceFunc)[0]
    cluster_new_id = np.argmin(SSE_cluster_new_list)
    cluster_new = cluster_new_list[cluster_new_id]
    # updating cluster list and centroid list
    cluster_list.append(cluster_new[0])
    cluster_list.append(cluster_new[1])
    centroid_list = FindCentroid(cluster_list, n_var)
  return cluster_list, centroid_list



