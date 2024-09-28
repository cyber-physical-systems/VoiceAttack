#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nilmtk


# In[2]:


from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")


# In[3]:


from nilmtk_contrib.disaggregate import DAE


# In[4]:


from nilmtk.disaggregate import CO
from nilmtk_contrib.disaggregate import DAE,Seq2Point, Seq2Seq, RNN, WindowGRU
from nilmtk.disaggregate import Hart85
from nilmtk.disaggregate import Mean


# In[5]:


from nilmtk_contrib.disaggregate import DAE
# from nilmtk.disaggregate import FHMM, Mean
from nilmtk.disaggregate import FHMMExact, Mean


# In[6]:


# redd = {
#   'power': {'mains': ['apparent','active'],'appliance': ['apparent','active']},
#   'sample_rate': 1,
#   'appliances': ['fridge'],
#    # 'methods': {"Mean":Mean({}),"FHMM_EXACT":FHMM_EXACT({'num_of_states':2}), "CombinatorialOptimisation":CO({})},
#     'methods': {   
#    # 'FHMM':FHMMExact({'num_of_states':3})
# #     'WindowGRU':WindowGRU({'n_epochs':20,'batch_size':64}),
#      'RNN':RNN({'n_epochs':30,'batch_size':64}),
# #      'DAE':DAE({'n_epochs':25,'batch_size':64}),
# #       'Seq2Point':Seq2Point({'n_epochs':25,'batch_size':64}),
# #       'Seq2Seq':Seq2Seq({'n_epochs':25,'batch_size':64}),
# #        'CO': CO({}),
# #       'Mean': Mean({}),
# #         "FHMM_EXACT":FHMM({'num_of_states':2}),
#     },
  
    
#   'train': {    
#     'datasets': {
#         'Dataport': {
#             'path': ' ',
#             'buildings': {
#                 2: {
#                     'start_time': '2024-04-23',
#                     'end_time': '2024-04-29'
#                     }
#                 }                
#             }
#         }
#     },
#   'test': {
#     'datasets': {
#         'Dataport': {
#             'path': ' ',
#             'buildings': {
#                 2: {
#                     'start_time': '2024-04-23',
#                     'end_time': '2024-04-29'
#                     }
#                 }
#             }
#         },
#         'metrics':['rmse']
#     }
# }

# api_results_experiment_1 = API(redd)
# # import pandas as pd

# import pandas as pd
# import os

# main = api_results_experiment_1.test_mains[0]
# gt_overall = api_results_experiment_1.gt_overall# Adjust based on actual keys if needed
# pred_overall_win = api_results_experiment_1.pred_overall['WindowGRU'] # Adjust based on actual keys if needed
# pred_overall_RNN = api_results_experiment_1.pred_overall['RNN']
# pred_overall_DAE = api_results_experiment_1.pred_overall['DAE']
# pred_overall_Seq2Point = api_results_experiment_1.pred_overall['Seq2Point'] 
# pred_overall_Seq2Seq = api_results_experiment_1.pred_overall['Seq2Seq']
# pred_overall_Mean = api_results_experiment_1.pred_overall['Mean'] 
# pred_overall_CO = api_results_experiment_1.pred_overall['CO']

# #Combine the ground truth and predictions into one DataFrame
# #Assuming that both DataFrames have the same index
# combined_df = pd.concat([main, gt_overall, pred_overall_win,pred_overall_RNN,pred_overall_DAE,pred_overall_Seq2Point,
#                          pred_overall_Seq2Seq,pred_overall_Mean,pred_overall_CO], axis=1, ignore_index=True,
#                         keys=['main','GT', 'win','RNN','DAE','Seq2Point','Seq2Seq','Mean','CO'])
# # Rename columns for clarity
# combined_df.columns = ['main','GT', 'win','RNN','DAE','Seq2Point','Seq2Seq','Mean','CO']
# # Saving the combined DataFrame to a CSV file

# path = ' '
# try:
#     os.makedirs(path, exist_ok=True)  # exist_ok=True will not raise an error if the directory already exists
#     print(f"Directory '{path}' was created successfully.")
# except OSError as error:
#     print(f"Creation of the directory '{new_folder_path}' failed due to: {error}")
    
# combined_df.to_csv(path + 'combined_results.csv')
# print("Combined DataFrame saved to 'combined_results.csv'.")

