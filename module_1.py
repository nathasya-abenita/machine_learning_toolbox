# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:51:18 2023

@author: User2021
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv

def InitializeData (df, degree):
  #Mengubah pandas menjadi numpy
  np_array = df.to_numpy()

  #Pengisian Matriks X dan T dalam bentuk matriks
  x = np.zeros((len(df),degree+1))
  t = np.zeros((len(df),1))

  #Pengisian matriks adalah per kolom dari atas hingga bawah
  for j in range (degree+1):
    for i in range (len(df)):
      if j == 0:
        x[i][j] = 1
      else:
        x[i][j] = np_array[i,0] ** j

  for i in range(len(df)):
    t[i][0] = np_array[i,1]

  return x, t

def LinearRegression (x, t, degree):
  #Perhitungan W topi di slide 14  ------------  T besar melambangkan transpose
  xT = np.transpose(x)
  xTx_inverse = inv(np.matmul(xT, x))
  xTx_inverse_xT = np.matmul(xTx_inverse, xT)
  w = np.matmul(xTx_inverse_xT, t)
  return w

def Predict (x_prediksi, degree, w):
  xnew = []
  for i in range (degree+1):
    xnew.append(x_prediksi**i)
  wT = np.transpose(w)
  x = np.transpose(xnew)
  return np.matmul(wT, x)

def CV_ArrPreparation (k, x, t, degree):
  # Mengatur jumlah datum pada tiap blok (membagi hampir sama rata)
  num, div = len(x), k
  block_size = [num // div + (1 if x < num % div else 0) for x in range (div)]

  x_training_list, t_training_list, w_list = [], [], []
  x_validation_list, t_validation_list = [], []

  block_size_count = 0

  for i in range (k):
    ##print("i ", i)
    x_new, t_new = x.copy(), t.copy()
    row_id_to_del = []
    for j in range (block_size_count, block_size_count + block_size[i]):
      ##print(j)
      # Adding rows for x_others, t_others
      if j == block_size_count:
        x_others, t_others = x[j:j+1, :], t[j:j+1, :]
      else:
        x_others = np.append(x_others, x[j:j+1, :], 0)
        t_others = np.append(t_others, t[j:j+1, :], 0)

      # Deleting rows for x_new, t_new
      row_id_to_del.append(j)

    # Updating counter for index
    block_size_count += block_size[i]

    # Deleting rows for x_new, t_new
    x_new = np.delete(x_new, row_id_to_del, 0)
    t_new = np.delete(t_new, row_id_to_del, 0)

    # Saving relevant vector
    x_validation_list.append(x_others); t_validation_list.append(t_others);
    x_training_list.append(x_new); t_training_list.append(t_new);

    # Akhirnya, w baru bisa ditentukan
    w = LinearRegression(x_new, t_new, degree)
    w_list.append(w)
  return x_training_list, t_training_list, x_validation_list, t_validation_list, w_list

def CV_ValidationLoss (k, x_list, t_list, w_list):
  loss_list = []
  for i in range (k):
    # Mengambil matriks x, t, w terkait
    x, t, w = x_list[i], t_list[i], w_list[i]

    # Menggunakan rumus loss function
    mat = np.subtract(t, np.matmul(x, w))
    loss = (1/len(x)) * np.matmul(np.transpose(mat), mat)
    loss_list.append(loss)

  # Rata-rata loss function untuk tiap blok adalah nilai akhir validation loss
  loss = sum(loss_list)/k
  return loss[0][0]

def TrainingLoss (x, t, w):
  # Menggunakan rumus loss function
  mat = np.subtract(t, np.matmul(x, w))
  loss = (1/len(x)) * np.matmul(np.transpose(mat), mat)
  return loss[0][0]
