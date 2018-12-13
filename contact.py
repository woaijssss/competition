import numpy as np
import pandas as pd

def contact_df(full_time, plc_df, sensor_df):
    plc_df['last_time'] = full_time - plc_df['csv_no'] * 5
    plc_df = plc_df.drop(['csv_no', 'time'], axis = 1)
    result = pd.concat([plc_df, sensor_df], axis = 1, join_axes = [plc_df.index])
    return result

if __name__ == '__main__':
    dataloc = 'C:/Users/lzy/Desktop/data/test'
    
    r1 = pd.read_csv(dataloc+'/result.csv')
    r2 = pd.read_csv(dataloc+'/result02.csv')
    r3 = pd.read_csv(dataloc+'/result03.csv')
    final = pd.concat([r1, r2, r3])
    final.to_csv(dataloc+'/final.csv', index=False, sep=',')
    '''
    plc_df = pd.read_csv(dataloc+'/plc.csv')
    sensor_df = pd.read_csv(dataloc+'/new03.csv')
    after_contact = contact_df(185, plc_df, sensor_df)
    after_contact.to_csv(dataloc+'/result03.csv', index=False, sep=',')
    '''

