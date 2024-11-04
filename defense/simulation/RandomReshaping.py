import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np


def run(hi):
    df = pd.read_csv('/home/keyangyu/PycharmProjects/PrivacyGuard/data/1dayflow/all.csv', header=None)
    d_col = ['Time', 'Size']
    df.columns = d_col
    df.index = range(len(df))

    x = df.index
    y1 = df['Size']
    plt.plot(x, y1, color='b')

    print(df)
    q = pd.Series([df['Size'].quantile(0.35), df['Size'].quantile(hi/100)])
    '''insert = df[df['Size'] > df['Size'].quantile(.85)]
    indexlist = []
    i = 0
    while i < len(insert) - 3:
        if (insert.index[i + 1] - insert.index[i] == 1) and (insert.index[i + 2] - insert.index[i + 1] == 1) \
                and (insert.index[i + 3] - insert.index[i + 2] == 1):
            indexlist.extend([insert.index[i], insert.index[i + 1], insert.index[i + 2], insert.index[i + 3]])
            i += 4
        else:
            i += 1
    print(indexlist)'''
    extend = 0
    for i in range(0, len(df)):
        '''if extend > 0:
            extend -= 1
            if df.iloc[i]['Size'] < df.iloc[i-1]['Size']:
                df.iloc[i]['Size'] = df.iloc[i - 1]['Size']
                continue'''
        '''boo = random.randint(0, 100)
        if boo > 99:
            ins = random.randint(0, len(insert) - 4)
            df.iloc[i - 3]['Size'] = insert.iloc[ins]['Size']
            df.iloc[i - 2]['Size'] = insert.iloc[ins + 1]['Size']
            df.iloc[i - 1]['Size'] = insert.iloc[ins + 2]['Size']
            df.iloc[i]['Size'] = insert.iloc[ins + 3]['Size']'''

        if (df.iloc[i]['Size'] > q[0]) and (df.iloc[i]['Size'] < q[1]):
            df.iloc[i]['Size'] = q[1]

        '''if df.iloc[i]['Size'] > q[1]:
            df.iloc[i]['Size'] = df.iloc[i]['Size'] * random.uniform(1.1, 1.3)
            extend = random.randint(3, 7)'''

    # df['Time'] = pd.to_datetime(df['Time'], unit='s')

    # df = df.resample('1min').sum()
    y2 = df['Size']
    plt.plot(x, y2, color='r')
    # plt.show()
    df.to_csv('/home/keyangyu/Desktop/PCC/' + str(hi) + '.csv', index=False, header=False)

    '''df = pd.read_csv('/home/keyangyu/PycharmProjects/PGOnline/CompareMCC/RandomReshaping/' + str(hi) + '.csv', header=None)
    col = ['Time', 'Size']
    df.columns = col
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.index = df['Time']
    df = df.drop(['Time'], axis=1)
    df = df.resample('1min').sum()
    print(df)
    df.to_csv('/home/keyangyu/PycharmProjects/PGOnline/CompareMCC/RandomReshaping/' + str(hi) + '.csv',header=False)'''



run(99)