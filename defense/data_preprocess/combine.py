import pandas as pd
import glob

input_path = ''
output_path = '1min.csv'

header = []

df = pd.read_csv(input_path + '0101' + '.csv')

# days = ['02','03','04','05','06','07','08','09','10']
days = ['0103','0104','0106','0107','0108','0109','0110','0111','0112']
for name in days:

    df1 = pd.read_csv(input_path + str(name) + '.csv')
    df = [df, df1]
    df = pd.concat(df)
       
df.to_csv(output_path , header=True, mode='a+')


    


    

