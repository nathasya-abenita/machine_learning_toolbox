# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:18:35 2024

@author: User2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Distance

def Euclidean (x, np_array):
  distance = np.zeros((len(np_array), 1))
  for i in range (len(np_array)):
    sum = 0
    for j in range (len(np_array[0])-1):
      sum += (x[j] - np_array[i][j])**2
    distance[i][0] = sum**(1/2)
  return distance

def Manhattan (x, np_array):
  distance = np.zeros((len(np_array), 1))
  for i in range (len(np_array)):
    sum = 0
    for j in range (len(np_array[0])-1):
      sum += np.absolute((x[j] - np_array[i][j]))
    distance[i][0] = sum
  return distance

def Hamming (x, np_array):
  np_array_lower = [[element.lower() for element in row] for row in np_array]
  x_lower = [word.lower() for word in x]
  distance = np.zeros((len(np_array), 1))
  for i in range (len(np_array)):
    sum = 0
    for j in range (len(np_array[0])-1):
      if x_lower[j] != np_array[i][j]:
        sum += 1
    distance[i][0] = sum
  return distance

## M
def PredictNN (k_nn, x, x_data_new, t_data_new, DistanceFunc): # ((k, x_new, t_new))
  np_array = np.concatenate((x_data_new, t_data_new), axis=1)
  distance = DistanceFunc(x, np_array)
  #Data adalah matriks yang sudah terisi dengan distance. Distance terletak pada kolom paling kanan
  data = np.zeros ((len(np_array), len(np_array[0])+1))
  for i in range (len(np_array)):
    data[i][len(np_array[0]+1)] = distance[i][0]
    for j in range (len(np_array[0])):
      data[i][j] = np_array[i][j]

  #Proses ini sedang sort data berdasarkan distance nya. Setelah di sort, disimpan data dalam data_sort_matrix
  data_sort = pd.DataFrame(data)
  data_sort.sort_values(len(np_array[0]), inplace = True)
  data_sort_matrix = data_sort.to_numpy()

  #Menyimpan semua class yang ada dalam variabel class_unique
  class_array = np_array[:,len(np_array[0])-1]
  class_unique = list(set(class_array))

  #Class_counter berguna untuk menghitung jumlah masing" kelas yang terdekat dengan input
  class_counter = np.zeros(len(class_unique))
  for i in range(len(class_unique)):
    for j in range (k_nn):
      if class_unique[i] == data_sort_matrix[j][len(np_array[0])-1]:
        class_counter[i] += 1

  #Menghitung kelas mana yang punya dataset counter terbesar
  predicted_class = 0
  highest_counter = 0
  for i in range (len(class_unique)):
    if class_counter[i] > highest_counter:
      highest_counter = class_counter[i]
      predicted_class = class_unique[i]

  return predicted_class


## 10-Fold CV Functions

def Loss_01 (true_list, predict_list):
  # Input
  # true_list : true class of each data
  # predict_list : predicted class for each data

  # Output
  # avg : average of loss
  count_loss = 0

  # Untuk tiap kelas yang salah diprediksi, counter loss bertambah 1
  for i in range (len(true_list)):
    if true_list[i][0] != predict_list[i][0]:
      count_loss += 1
  avg = count_loss/len(true_list)
  return avg

def CV_ValidationLoss (t_validation_list, t_new_list):
  loss_list = []
  for i in range (10):
    # Mengambil matriks x, t, w terkait
    t, t_new = t_validation_list[i], t_new_list[i]

    # Menggunakan rumus loss function
    loss = Loss_01 (t, t_new)
    loss_list.append(loss)

  # Rata-rata loss function untuk tiap blok adalah nilai akhir validation loss
  loss = sum(loss_list)/10
  return loss

def CV_MinimizingLoss (k_max, x_data, t_data, DistanceFunc):
  # Output
  # Nilai k yang meminimumkan validation loss

  loss_list = []
  for k in range (1, k_max+1): # range of k to be checked
    print("checking k:", k)
    t_validation_list, t_new_list = CV_ArrPreparation (k, x_data, t_data, DistanceFunc)
    loss_list.append(CV_ValidationLoss (t_validation_list, t_new_list))

  # Output
  k_nn_optimal = np.argmin(loss_list)+1 # Ditambah 1 karena k terkecil adalah 1
  return k_nn_optimal, loss_list

def CV_ArrPreparation (k, x_data, t_data, DistanceFunc):
  # Input
  # x_data : matrix with each row represents attributes of a data, number of columns represents number of attribute
  # t_data : true class for each data

  # Output
  # t_validation_list : t_others (kelas-kelas sebenarnya) untuk tiap fold
  # t_predict_list : t_predict (kelas-kelas hasil prediksi) untuk tiap fold

  # Mengatur jumlah datum pada tiap blok (membagi hampir sama rata)
  num, div = len(x_data), 10
  block_size = [num // div + (1 if i < num % div else 0) for i in range (div)]

  x_training_list, t_training_list, t_predict_list = [], [], []
  x_validation_list, t_validation_list = [], []

  block_size_count = 0

  for i in range (10):
    x_new, t_new = x_data.copy(), t_data.copy()
    row_id_to_del = []
    for j in range (block_size_count, block_size_count + block_size[i]):
      # Adding rows for x_others, t_others
      if j == block_size_count:
        x_others, t_others = x_data[j:j+1, :], t_data[j:j+1, :]
      else:
        x_others = np.append(x_others, x_data[j:j+1, :], 0)
        t_others = np.append(t_others, t_data[j:j+1, :], 0)

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

    # Akhirnya, predicted class untuk data x validasi bisa ditentukan
    t_predict = np.zeros((len(t_others),1))
    for j in range (len(t_predict)):
      t_predict[j] = PredictNN(k, x_others[j], x_new, t_new, DistanceFunc)
    t_predict_list.append(t_predict)
    #print(t_others.shape, "|", t_predict.shape)
  return t_validation_list, t_predict_list


## Confusion Matrix

def ConfusionMatrix (c, t_data, t_new_list):
  # c : number of class
  # t_data : list of t
  # t_new_list : list of t_new
  # t : true class for each data
  # t_new : predicted class for each data

  mat = np.zeros((c,c))
  for i in range (len(t_data)):
    t = int(t_data[i])
    t_new = int(t_new_list[i])
    mat[t_new][t] += 1
  return mat