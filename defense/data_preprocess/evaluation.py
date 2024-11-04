import pandas as pd
import numpy as np
import os
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_absolute_error
from math import sqrt
metric = {}
met = 'RMSE'
# threshold = 400000
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def Mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) 
house_num = 5
# models= ["CO",'RNN','WindowGRU','DAE','Seq2Seq','Mean']
models= ["FHMM"]


for model in models:

    rmse = 0
    mape = 0 
    sum_  = 0
    rmse ={}
    mape = {}
    mae = {}
    input_path =  '' 
    output_path = 'mape.csv'

#     isExist = os.path.exists(output_path)
#     if not isExist:
#         os.mkdir(output_path)
    for i in [3,4,5]:
        groundtruth = pd.read_csv(input_path + 'groundtruth.csv')
        prediction = pd.read_csv(input_path + str(model) +  '.csv')
        gt_ = np.array(groundtruth.iloc[:,[i]])
        pred_ = np.array(prediction.iloc[:,[i]])
        #
        rmse[i] =  sqrt(mean_squared_error(gt_,pred_))
        mape[i] =  Mean_absolute_percentage_error(gt_,pred_)
    with open(output_path , 'a') as f:
        writer = csv.writer(f)
        writer.writerow([model,mape[3],mape[4],mape[5]])  

    
    

#     for i in range(2,house_num-2):
#     for i in range(1,house_num+1):
#     for i in range(1,11):
# #     for i in [1,3,4,5,6,7,8,14,16,18]:
# #     for i in [1,2,3,4,5]:
#         gt_ = np.array(groundtruth.iloc[:,[i]])
#         pred_ = np.array(prediction.iloc[:,[i]])
# #
#         rmse[i] =  sqrt(mean_squared_error(gt_,pred_))
#         mape[i] =  Mean_absolute_percentage_error(gt_,pred_)
# #         mae[i] = mean_absolute_error(gt_,pred_)
#     with open(output_path , 'a') as f:
#         writer = csv.writer(f)
# #         writer.writerow([model,mape[1],mape[3],mape[4],mape[5],mape[6],mape[7],mape[8],mape[14],mape[16], mape[18]]) 
#         writer.writerow([model,mape[1],mape[2],mape[3],mape[4],mape[5],mape[6],mape[7],mape[8],mape[9],mape[10]])  

