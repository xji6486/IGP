#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 14:06:28 2018

@author: chunhsuanlojason
"""
import os,sys
import copy, numpy as np
 
np.random.seed(0)
# sigmoid 函數
def sigmoid(x):
 return 1/(1+np.exp(-x))
# sigmoid 導函數
def dlogit(output): # dlogit
 return output*(1-output)
# 十進位轉二進位數字組
def int2binary(bindim,largest_number):
 int2bindic = {}
 binary = np.unpackbits(np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
 for i in range(largest_number):
  int2bindic[i] = binary[i]
 return int2bindic 
# 樣本發生器:實現一個簡單的(a + b = c)的加法
def gensample(dataset,largest_number):
  # 實現一個簡單的(a + b = c)的加法
  a_int = np.random.randint(largest_number/2.5) # 十進位 產生0-256/2.5 之間的亂數
  a = dataset[a_int] # 二進位  
  b_int = np.random.randint(largest_number/3) # 十進位 產生0-256/3 之間的亂數
  b = dataset[b_int] # 二進位    
  c_int = a_int + b_int # 十進位的結果
  c = dataset[c_int] # 十進位轉二進位的結果
  return a,a_int,b,b_int,c,c_int
  
def showresult(j,overallError,d,c,a_int,b_int):
 if(j % 1000 == 0):
  print ("Error:" + str(overallError))
  print ("Pred:" + str(d))
  print ("True:" + str(c))
  out = 0
  for index,x in enumerate(reversed(d)):
   out += x*pow(2,index)
  print (str(a_int) + " + " + str(b_int) + " = " + str(out))
  print ("------------") 
#1. 產生的訓練集樣本
binary_dim= 8 # 生成的二進位bitset的寬度
largest_number = pow(2,binary_dim) # 最大數2^8=256
dataset = int2binary(binary_dim,largest_number) # 產生資料集
 
#2. 初始化網路參數
alpha = 0.1 # 學習速率
input_dim = 2 # 輸入神經元個數
hidden_dim = 16 # 隱藏層神經元個數
output_dim = 1 # 輸出神經元個數
maxiter = 10000 # # 最大反覆運算次數
 
# 初始化 LSTM 神經網路權重 synapse是神經元突觸的意思
synapse_I = 2*np.random.random((input_dim,hidden_dim)) - 1 # 連接了輸入層與隱含層的權值矩陣
synapse_O = 2*np.random.random((hidden_dim,output_dim)) - 1 # 連接了隱含層與輸出層的權值矩陣
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1 # 連接了隱含層與隱含層的權值矩陣
# 權值更新緩存：用於存儲更新後的權值。np.zeros_like：返回全零的與參數同類型、同維度的陣列
synapse_I_update = np.zeros_like(synapse_I) # 
synapse_O_update = np.zeros_like(synapse_O) # 
synapse_h_update = np.zeros_like(synapse_h) # 
 
#3. 主程序--訓練過程：
for j in range(maxiter): 
 # 在實際應用中，可以從訓練集中查詢到一個樣本: 生成形如[a] [b]--> [c]這樣的樣本：
 a,a_int,b,b_int,c,c_int = gensample(dataset,largest_number)
  
 # 初始化一個空的二進位數字組，用來存儲神經網路的預測值
 d = np.zeros_like(c)
  
 overallError = 0 # 重置全域誤差
 
 layer_2_deltas = list(); # 記錄layer 2的導數值
 layer_1_values = list(); # 與layer 1的值。
 layer_1_values.append(np.zeros(hidden_dim)) # 初始化時無值，存儲一個全零的向量
  
 # 正向傳播過程：逐個bit位(0,1)的遍歷二進位數字字。
 for position in range(binary_dim): 
  indx = binary_dim - position - 1 # 陣列索引7,6,5,...,0
  # X 是樣本集的記錄，來自a[i]b[i]; y是樣本集對應的標籤,來自c[i]
  X = np.array([[a[indx],b[indx]]])
  y = np.array([[c[indx]]]).T
   
  # 隱含層 (input ~+ prev_hidden)
  # 1. np.dot(X,synapse_I)：從輸入層傳播到隱含層：輸入層的資料*（輸入層-隱含層的權值）
  # 2. np.dot(layer_1_values[-1],synapse_h)：從上一次的隱含層[-1]到當前的隱含層：上一次的隱含層權值*當前隱含層的權值
  # 3. sigmoid(input + prev_hidden)
  layer_1 = sigmoid(np.dot(X,synapse_I) +np.dot(layer_1_values[-1],synapse_h))
    
  # 輸出層 (new binary representation)
  # np.dot(layer_1,synapse_O)：它從隱含層傳播到輸出層，即輸出一個預測值。
  layer_2 = sigmoid(np.dot(layer_1,synapse_O))
    
  # 計算預測誤差
  layer_2_error = y - layer_2
  layer_2_deltas.append((layer_2_error)*dlogit(layer_2)) # 保留輸出層每個時刻的導數值
  overallError += np.abs(layer_2_error[0]) # 計算二進位位元的誤差絕對值的總和，標量
     
  d[indx] = np.round(layer_2[0][0]) # 存儲預測的結果--顯示使用
   
  layer_1_values.append(copy.deepcopy(layer_1)) # 存儲隱含層的權值，以便在下次時間反覆運算中能使用
   
 future_layer_1_delta = np.zeros(hidden_dim) # 初始化下一隱含層的誤差
 # 反向傳播：從最後一個時間點開始，反向一直到第一個： position索引0,1,2,...,7
 for position in range(binary_dim):   
  X = np.array([[a[position],b[position]]]) 
   
  layer_1 = layer_1_values[-position-1] # 從列表中取出當前的隱含層。從最後一層開始，-1，-2，-3
  prev_layer_1 = layer_1_values[-position-2] # 從列表中取出當前層的前一隱含層。
     
  layer_2_delta = layer_2_deltas[-position-1] # 取出當前輸出層的誤差
  # 計算當前隱含層的誤差:
  # future_layer_1_delta.dot(synapse_h.T): 下一隱含層誤差*隱含層權重
  # layer_2_delta.dot(synapse_O.T):當前輸出層誤差*輸出層權重
  # dlogit(layer_1)：當前隱含層的導數
  layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) +layer_2_delta.dot(synapse_O.T)) *dlogit(layer_1)
    
  # 反向更新權重: 更新順序輸出層-->隱含層-->輸入層
  # np.atleast_2d：輸入層reshape為2d的陣列
  synapse_O_update +=np.atleast_2d(layer_1).T.dot(layer_2_delta)
  synapse_h_update +=np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
  synapse_I_update += X.T.dot(layer_1_delta)
  
  future_layer_1_delta = layer_1_delta # 下一隱含層的誤差
 # 更新三個權值 
 synapse_I += synapse_I_update * alpha
 synapse_O += synapse_O_update * alpha
 synapse_h += synapse_h_update * alpha 
 # 所有權值更新項歸零
 synapse_I_update *= 0; synapse_O_update *= 0; synapse_h_update *= 0
  
 # 逐次列印輸出
 showresult(j,overallError,d,c,a_int,b_int)
