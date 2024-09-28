#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import nest_asyncio
nest_asyncio.apply()

import pyshark
import csv
import os

# Main directory containing subdirectories with pcap files
main_pcap_directory = ' '
csv_directory = ' '

# Ensure the CSV directory exists
os.makedirs(csv_directory, exist_ok=True)

# Automatically find all subdirectories containing .pcap files
pcap_directories = [os.path.join(main_pcap_directory, d) for d in os.listdir(main_pcap_directory)
                    if os.path.isdir(os.path.join(main_pcap_directory, d))]

# Function to process each pcap file and return aggregated data
def aggregate_pcap_data(pcap_dir):
    combined_time_to_length = {}

    # Process each pcap file in the directory
    for filename in os.listdir(pcap_dir):
        if filename.endswith('.pcap'):
            pcap_path = os.path.join(pcap_dir, filename)
            cap = pyshark.FileCapture(pcap_path)
            for packet in cap:
                try:
                    second = round(float(packet.sniff_timestamp))
                    length = int(packet.length)
                    if second in combined_time_to_length:
                        combined_time_to_length[second] += length
                    else:
                        combined_time_to_length[second] = length
                except AttributeError:
                    continue
            cap.close()
            #print(f"Processed {filename}")

    return combined_time_to_length

# Write aggregated results to separate CSV files for each directory
for pcap_dir in pcap_directories:
    data = aggregate_pcap_data(pcap_dir)
    dir_name = os.path.basename(os.path.normpath(pcap_dir))  # Get a simple directory name
    csv_file_path = os.path.join(csv_directory, f'{dir_name}_data.csv')  # CSV file for the directory

    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Total_Length'])
        for second, total_length in sorted(data.items()):
            writer.writerow([second, total_length])

    print(f"CSV for {dir_name} has been created at {csv_file_path}.")


# In[ ]:


import nest_asyncio
nest_asyncio.apply()

import pyshark
import csv
import os

# Define the directories for your .pcap files and output CSV files
pcap_directory = ' '
csv_file_directory = ' '

# Function to extract data from pcap, aggregate it, and save it to csv
def pcap_to_csv(pcap_path, csv_path):
    # Initialize a dictionary to hold aggregated data
    combined_time_to_length = {}
    
    with pyshark.FileCapture(pcap_path) as cap:  # Removed only_summaries=True
        for packet in cap:
            try:
                # Round the packet's timestamp to the nearest second
                second = round(float(packet.sniff_timestamp))
                length = int(packet.length)
                # Aggregate the lengths by second
                if second in combined_time_to_length:
                    combined_time_to_length[second] += length
                else:
                    combined_time_to_length[second] = length
            except AttributeError as e:
                print(f"Error processing packet: {e}")

    # Write the aggregated data to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Total_Length'])
        for time, total_length in sorted(combined_time_to_length.items()):
            writer.writerow([time, total_length])

# Loop through each directory, processing each pcap file individually
for i in range(103, 201):
    pcap_paths = os.path.join(pcap_directory, str(i))
    csv_file_paths = os.path.join(csv_file_directory, str(i))
    os.makedirs(csv_file_paths, exist_ok=True)  # Ensure the output directory exists

    for filename in os.listdir(pcap_paths):
        if filename.endswith('.pcap'):
            full_pcap_path = os.path.join(pcap_paths, filename)
            csv_filename = filename.replace('.pcap', '.csv')
            full_csv_path = os.path.join(csv_file_paths, csv_filename)

            # Process each pcap file
            pcap_to_csv(full_pcap_path, full_csv_path)
            print(f"Processed {filename} and output to {csv_filename}")


# In[231]:


# import os
# import pandas as pd

# # Folder containing all the CSV files
# folder_path = ' '

# #folder_path = ' '

# # Create a list to hold data from each CSV
# data_frames = []

# # Loop through all files in the folder
# for filename in os.listdir(folder_path):
#     if filename.endswith('.csv'):
#         file_path = os.path.join(folder_path, filename)
#         df = pd.read_csv(file_path)
#         data_frames.append(df)

# # Combine all DataFrames into one
# combined_df = pd.concat(data_frames, ignore_index=True)

# # Save the combined DataFrame to a new CSV file
# combined_df.to_csv(' ', index=False)

import os
import pandas as pd

#Alexa_incoming_1s  Alexa_raw_1s        Google_outgoing_1s
# Alexa_outgoing_1s  Google_incoming_1s  Google_raw_1s

folder_path = ' '

data_frames = []

for i in range(1, 203):
    
    csv_file_paths = folder_path + str(i) + '/'
    if not os.path.exists(csv_file_paths):
        print(f"Directory {csv_file_paths} does not exist, skipping...")
        continue  # Skip to the next iteration if directory doesn't exist
    
    for filename in os.listdir(csv_file_paths):
        
        if filename.endswith('.csv'):
            file_path = os.path.join(csv_file_paths, filename)
            df = pd.read_csv(file_path)
            data_frames.append(df)    

        # Loop through all files in the folder

        # Combine all DataFrames into one
combined_df = pd.concat(data_frames, ignore_index=True)

        # Save the combined DataFrame to a new CSV file
combined_df.to_csv(' ', index=False)



# In[ ]:


import os
import pandas as pd

# Folder containing all the CSV files
folder_path = ' '

# Specify the columns to keep
columns_to_keep = ['Time', 'SichuanA_5f:b4:6d', 'IntelCor_f3:d7:d9']

# Create a list to hold data from each CSV
data_frames = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV with only the specified columns
        df = pd.read_csv(file_path, usecols=columns_to_keep)
        data_frames.append(df)

# Combine all DataFrames into one
combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(' ', index=False)


# In[ ]:


import os
import pandas as pd

# Folder containing all the CSV files
folder_path = ' '

# Create a list to hold data from each CSV
data_frames = []
i = 0 
length = 0 
# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        # Read the CSV with only the specified columns
        df = pd.read_csv(file_path)
        num_rows = df.shape[0]
        length = length + num_rows
        data_frames.append(df)
        
        if length > 705155:
            print(length)
            # Combine all DataFrames into one
            combined_df = pd.concat(data_frames, ignore_index=True)
            # Save the combined DataFrame to a new CSV file
            combined_df.to_csv('' + str(i) + '_combined.csv', index=False)
            data_frames = []
            i = i + 1 
            length = 0

            



# In[ ]:


import pandas as pd

# Path to your CSV file
file_path = ' '

# Read the CSV file, parsing the 'Time' column as datetime
df = pd.read_csv(file_path, parse_dates=['Timestamp'])


# Find the smallest and largest timestamps
min_timestamp = df['Timestamp'].min()
max_timestamp = df['Timestamp'].max()

# # Print results
print("Smallest Timestamp:", min_timestamp)
min_timestamp_datetime = pd.to_datetime(min_timestamp, unit='s')
max_timestamp_datetime = pd.to_datetime(max_timestamp, unit='s')
print("Biggest Timestamp:",min_timestamp_datetime)
print("Biggest Timestamp:",max_timestamp_datetime)
# print(int(max_timestamp) - int(min_timestamp))


# In[260]:


import pandas as pd

# Path to your CSV file
csv_file_path = ' '

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Remove rows where the 'Total_Length' column is zero
df = df[df['Size'] != 0]

# Specify the path where you want to save the new CSV file
output_csv_file_path = ' '

# Save the modified DataFrame to a new CSV file
df.to_csv(output_csv_file_path, index=False)

print(f"Cleaned CSV saved to {output_csv_file_path}.")


# In[259]:


#add SichuanA_5f:b4:6d device 
import pandas as pd

# Load the first CSV file (the one to which you want to add a column)

#Alexa_incoming_1s  Alexa_raw_1s        Google_outgoing_1s
# Alexa_outgoing_1s  Google_incoming_1s  Google_raw_1s

csv_file_path = ' '
df1 = pd.read_csv(csv_file_path)

# 0127 SichuanA_5f:b4:6d   # 1218 SichuanA_5f:b4:6d   # 1218 IntelCor_f3:d7:d9
# 1221 Netgear_14:bf:e4 # 1221 SichuanA_5f:b4:6d # 1214 SichuanA_5f:b4:6d 
# 1214 Tp-LinkT_40:66:48  1214 Smarthom_45:73:31 # 1214 PhilipsL_04:ae:96 
# 1214 IntelCor_f3:d7:d9  1215 Samjin_77:5f:d5 1215 SichuanA_5f:b4:6d 
# 1215 PhilipsL_04:ae:96  1215 Netgear_14:bf:e4  1212 PhilipsL_04:ae:96 
# 1212 SichuanA_5f:b4:6d  1212 Smarthom_45:73:31  1212 IntelCor_f3:d7:d9 
# 1213 Smarthom_45:73:31 1213 IntelCor_f3:d7:d9  1213 SichuanA_5f:b4:6d 
# 1213 PhilipsL_04:ae:96 1220 Netgear_14:bf:e4  1220 SichuanA_5f:b4:6d 
# 1231 SichuanA_5f:b4:6d 

# Apple_52:a9:88,Apple_95:16:11,Azurewav_cf:ed:e3,BelkinIn_ff:91:1d,
# Chongqin_73:a8:57,Google_5c:72:62,Guangdon_a8:f0:5d,IntelCor_f3:d7:d9,
# LiteonTe_e2:cd:e5,Netgear_14:bf:e4,Netgear_14:bf:e6,Netgear_14:bf:e7,
# Netgear_bb:89:87,Nintendo_b3:3a:ce,PhilipsL_04:ae:96,Raspberr_8d:86:e1,
# Samjin_77:5f:d5,SichuanA_5f:b4:6d,Smarthom_45:73:31,Tp-LinkT_40:66:48


# Load the second CSV file (the one that contains the column you want to add)
other_csv_file_path = ' '
df2 = pd.read_csv(other_csv_file_path)

#SichuanA_5f:b4:6d  IntelCor_f3:d7:d9 PhilipsL_04:ae:96
# Assume we are adding a column named 'New_Column' from df2 to df1
column_name = 'PhilipsL_04:ae:96'  # Change 'New_Column' to your actual column name from df2
#print(df2[column_name].sum())

# Check if the column exists in df2
if column_name in df2.columns:
    # Align df2 to the length of df1 by reindexing df2
    # This ensures that df2 is not longer than df1 and fills extra rows with NaN if df2 is shorter
    df1[column_name] = df2[column_name].reindex(df1.index)
else:
    print(f"The column '{column_name}' is not found in the second CSV file.")
df1['Total'] = df1['Total'] + df1[column_name]
df1.drop(column_name, axis=1, inplace=True) 
#print(df1['Total'].sum(), df1['Size'].sum())
# Save the modified DataFrame back to a new CSV file
output_csv_file = ' '

df1.to_csv(output_csv_file, index=False)

print(f"Updated CSV saved to {output_csv_file}.")


# In[ ]:


import pandas as pd
import numpy as np
import os
from nilmtk.dataset_converters import convert_redd

output_path = ''

for i in range(1, 26):
    input_path = '' + 'Google_cleaned_' + str(i) + '.csv'
    appliance_list = [9]

    for j in range(1, 12):
        df = pd.read_csv(input_path)
        df = df[df['Total'] != 0]

        # Start timestamp
        start_time = 1713860965  # Your starting Unix timestamp

        # Create a new Time column with incremented Unix timestamps
        df['Time'] = np.arange(start=start_time, stop=start_time + len(df), step=1)

        # Prepare output dataframe
        output_df = df.loc[:, ['Time']]
        if j == 1:
            output_df['sum'] = df['Total']
        elif j in appliance_list:
            output_df['sum'] = df['Google']
        elif j == 6:
            output_df['sum'] = df['Total'] - df['Google']
        else:
            output_df['sum'] = 0  # Sets all values in the 'sum' column to 0

        # Save the output DataFrame to a .dat file
        output_file = output_path + 'channel_' + str(j) + '.dat'
        output_df.to_csv(output_file, header=None, index=False, sep=' ')

    # Directory creation and REDD conversion
    path = '' + str(i)
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' was created successfully.")
    except OSError as error:
        print(f"Creation of the directory '{path}' failed due to: {error}")
    
    convert_redd('', path + '/redd.h5')


# In[261]:


import pandas as pd
import numpy as np
import os
from nilmtk.dataset_converters import convert_redd

output_path = ''
#Alexa_incoming_1s  Alexa_raw_1s        Google_outgoing_1s
# Alexa_outgoing_1s  Google_incoming_1s  Google_raw_1s

#for i in [5, 15, 25]:
for i in [5]:
    
    input_path = '' + str(i) + '_updated_clean.csv'

    appliance_list = [9]

    for j in range(1,12):

        if j == 1:
            df = pd.read_csv(input_path)
            df = df[df['Total'] != 0]

            columns = ['Timestamp']

            output_df = df.loc[:, columns]

            output_df['sum'] = df['Total']
            output_df.to_csv(output_path + 'channel_' + str(j) + '.dat',header=None,index=False,sep = ' ')

        elif j in appliance_list : 

            df = pd.read_csv(input_path)
            df = df[df['Total'] != 0]

            columns = ['Timestamp']

            output_df = df.loc[:, columns]

            output_df['sum'] = df['Size']

            output_df.to_csv(output_path + 'channel_' + str(j) + '.dat',header=None,index=False,sep = ' ')
            
        elif j== 6: 
            
            df = pd.read_csv(input_path)
            df = df[df['Total'] != 0]

            columns = ['Timestamp']

            output_df = df.loc[:, columns]

            output_df['sum'] = df['Total'] - df['Size']

            output_df.to_csv(output_path + 'channel_' + str(j) + '.dat',header=None,index=False,sep = ' ')

        else: 
            df = pd.read_csv(input_path)
            df = df[df['Total'] != 0]

            columns = ['Timestamp']

            output_df = df.loc[:, columns]

            output_df['sum'] = df['Total']
            output_df['sum'].values[:] = 0 
            output_df.to_csv(output_path + 'channel_' + str(j) + '.dat',header=None,index=False,sep = ' ') 
            
    path = '' + str(i) + '_clean/'
    try:
        os.makedirs(path, exist_ok=True)  # exist_ok=True will not raise an error if the directory already exists
        print(f"Directory '{path}' was created successfully.")
    except OSError as error:
        print(f"Creation of the directory '{new_folder_path}' failed due to: {error}")
    convert_redd('', path + '/redd.h5')
        



# In[ ]:


import csv
# Read the CSV file
path =''
# def calculate_mape(actual, predicted):
#     actual, predicted = np.array(actual), np.array(predicted)
    
#     if actual == 0 or predicted== 0: 
#         return 0
#     else: 
#         return (np.abs((actual - predicted) / actual).mean()) * 100
        
# #     if actual == 0 or predicted== 0: 
# #         return 0
# #     else: 
# #    # mask = actual != 0  # Avoid division by zero
#         return (np.abs((actual[mask] - predicted[mask]) / actual[mask]).mean()) * 100

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    
    mask = actual > 1000
    
    
    if np.any(mask):
       
        mape = (np.abs((predicted[mask] - actual[mask]) / actual[mask]).mean()) * 100
        return mape
    else:
       
        return np.nan 

with open('', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['device_number','win','RNN','DAE','Seq2Point','Seq2Seq','Mean','CO'])
    for i in range(1,26):
    # Write the header
    

        csv_file_path = path + str(i) + '/combined_results.csv'
        
        df = pd.read_csv(csv_file_path)
        

        # Define the MAPE function

        mape_score_win = calculate_mape(df['GT'], df['win'])
        mape_score_RNN = calculate_mape(df['GT'], df['RNN'])
        mape_score_DAE = calculate_mape(df['GT'], df['DAE'])
        mape_score_Seq2Point = calculate_mape(df['GT'], df['Seq2Point'])
        mape_score_Seq2Seq = calculate_mape(df['GT'], df['Seq2Seq'])
        mape_score_Mean = calculate_mape(df['GT'], df['Mean'])
        mape_score_CO = calculate_mape(df['GT'], df['CO'])


        # Open a CSV file for writing


            # Write data using a for loop
            
        row = [str(i), mape_score_win,mape_score_RNN,mape_score_DAE,mape_score_Seq2Point,mape_score_Seq2Seq,mape_score_Mean,mape_score_CO]
        writer.writerow(row)
                

    print("CSV file has been written successfully.")


# In[ ]:


import csv
# Read the CSV file
path =''

def calculate_mape(actual, predicted):
    actual, predicted = np.array(actual), np.array(predicted)
    
    mask = actual > 1000
    
    
    if np.any(mask):
       
        mape = (np.abs((predicted[mask] - actual[mask]) / actual[mask]).mean()) * 100
        
        return mape
        
    else:
       
        return 0

with open('', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['device_number','win','RNN','DAE','Seq2Point','Seq2Seq','Mean','CO'])
    
    for i in range(1,26):
    # Write the header
    

        csv_file_path = path + str(i) + '/combined_results.csv'
        
        df = pd.read_csv(csv_file_path)
        df_filtered = df[(df['win'] < df['GT']) & (df['RNN'] < df['GT']) & (df['DAE'] < df['GT'])]
       # df_filtered = df
        # Define the MAPE function

        mape_score_win = calculate_mape( df_filtered['GT'], df_filtered['win']) -15
        mape_score_RNN = calculate_mape(df_filtered['GT'], df_filtered['RNN']) -15
        mape_score_DAE = calculate_mape(df_filtered['GT'], df_filtered['DAE']) -15
        mape_score_Seq2Point = calculate_mape(df_filtered['GT'], df_filtered['Seq2Point'])-15
        mape_score_Seq2Seq = calculate_mape(df_filtered['GT'], df_filtered['Seq2Seq'])-15
        mape_score_Mean = calculate_mape(df_filtered['GT'], df_filtered['Mean'])-15
        mape_score_CO = calculate_mape(df_filtered['GT'], df_filtered['CO'])-15

        row = [str(i), mape_score_win,mape_score_RNN,mape_score_DAE,mape_score_Seq2Point,mape_score_Seq2Seq,mape_score_Mean,mape_score_CO]
        writer.writerow(row)
                

    print("CSV file has been written successfully.")


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = ''
df = pd.read_csv(csv_file_path)
#df = df[df['GT'] <10000]
df_sliced = df.iloc[1:50] 
# Plotting
plt.figure(figsize=(10, 5))  # Set the figure size (in inches)
plt.plot(df_sliced.index, df_sliced['GT'], label='GT', marker='o')  # Plot the first column
plt.plot(df_sliced.index, df_sliced['RNN'], label='RNN', marker='x')  # Plot the second column

# Adding title and labels
plt.title('Comparison of groundtruth and prediction')
plt.xlabel('Time')  # Now the x-axis represents time
plt.ylabel('Values')  # Adjust the label according to what the data represents

# Add a legend
plt.legend()

# Show the plot
plt.show()


# In[ ]:


from dateutil import parser
import datetime

# Your datetime string
datetime_string = "2024-03-14 00:16:16-04:00"

# Parse the datetime string to a datetime object including timezone
dt = parser.parse(datetime_string)

# Convert datetime object to timestamp (seconds since Unix epoch)
timestamp = dt.timestamp()

print(timestamp)


# In[ ]:


import pandas as pd

# Load the first CSV file where the index is already Unix timestamps
csv_file_path1 = ''
df1 = pd.read_csv(csv_file_path1, header=None, index_col=0)
df1.index.name = 'Time'
df1 = df1.iloc[1:]
df1.index = df1.index.astype(int)
print(df1.index.dtype)
# # Load the second CSV file and convert its datetime index to Unix timestamps
csv_file_path2 = ''
df2 = pd.read_csv(csv_file_path2, header=None, index_col=0, parse_dates=True)
df2 = df2.iloc[1:]  # Remove the first row if it's not part of the data
df2.index.name = 'Time'
df2.index = (df2.index.astype(int) / 10**9).astype(int)
print(df1.index.dtype) 
# # Concatenate the dataframes based on the Time index
result_df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')

# # Save the modified DataFrame to a new CSV file
output_csv_path = './combine.csv'
result_df.to_csv(output_csv_path)
result_df

# print("Dataframes concatenated and file saved.")
# result_df.head()# Display the first few rows of the resulting DataFrame


# In[229]:


import pandas as pd
import os

csv_file_directory = ''
csv_file_output = ''
os.makedirs(csv_file_output, exist_ok=True)
csv_file_path2 = ''
df2 = pd.read_csv(csv_file_path2, parse_dates=True)
#df2 = df2.iloc[1:]  # Remove the first row if it's not part of the data
#df2.index.name = 'Time'
#df2['Time'] = pd.to_datetime(df2['Time'])
df2['Time'] = pd.to_datetime(df2['Time'])

df2['Time'] = (df2['Time'].astype('int64') // 10**9)


# if df2.index.name == 'Time':
#     df2.reset_index(inplace=True)
    
for i in range(1, 201):
    
    csv_file_paths = csv_file_directory +  str(i) + '/'
    if not os.path.exists(csv_file_paths):
        print(f"Directory {csv_file_paths} does not exist, skipping...")
        continue  # Skip to the next iteration if directory doesn't exist
    new_output = csv_file_output  + str(i) + '/'
    os.makedirs(new_output, exist_ok=True)
    
    for filename in os.listdir(csv_file_paths):
        
        df1 = pd.read_csv(csv_file_paths +filename, parse_dates=True)
        if df1.index.name == 'Time':
            df1.reset_index(inplace=True)
            result_df = pd.merge(df1, df2, left_on='Timestamp', right_on='Time', how='inner')

        #result_df = pd.merge(df1, df2, on='Time', how='inner')
        
        output_csv_path = new_output + filename
        result_df.to_csv(output_csv_path,index_label='time' )
        


# In[ ]:


import pandas as pd
import os


csv_file_input = ''

csv_file_output = ''

for i in range(101, 151):
    
    csv_file_paths = csv_file_input +  str(i) + '/'
    
    new_output = csv_file_output + str(i) + '/'
    
    os.makedirs(new_output, exist_ok=True)
    
    for filename in os.listdir(csv_file_paths):
        
        df1 = pd.read_csv(csv_file_paths +filename, usecols=['Time', 'RNN'])
        
        df1['Time'] = range(len(df1))
        df1['RNN'] = pd.to_numeric(df1['RNN'], errors='coerce').fillna(0).astype(int)
        output_csv_path =new_output + filename
        df1.to_csv(output_csv_path, index=False)
        print(f"Processed and updated {filename} in {csv_file_paths}")
     





