
import src.samplingSensorData as samplingSensorData
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess

import logging

'''
    整测试集sensor数据：
        包括去除异常值、inf值
        去矢量化、取平均值等
'''

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                # filename='./record.log',
                filemode='w')

if __name__ == '__main__':
    dp = dataset_preprocess.DataSetPreprocess()
    
    for i in range(1, 4):
        plc_csv_dir = '../../01-TrainingData-qLua/0' + str(i)        # 训练集
        # plc_csv_dir = '../../02-TestingData-poL3/0' + str(i)        # 测试集
        filename = plc_csv_dir + '/PLC/plc_test.csv'
        columns = ['vibration_1', 'vibration_2', 'vibration_3', 'current']
        # df_new = DataFrame(columns=columns)
        arr = []
        
        # ssd = samplingSensorData.SamplingSensorData(plc_csv_dir, test_filename=filename)  # 读取plc.csv文件
        ssd = samplingSensorData.SamplingSensorData(plc_csv_dir)  # 读取plc.csv文件
        
        logging.debug('第 %d 个plc文件:', i)
        
        for line in range(0, ssd.getPLCLength()):  # 对当前plc所有数据操作
            new_sensor_df = ssd.samplingSensorData(line=line)  # 计算后的结果
            
            if not new_sensor_df.empty:
                arr.append(list(new_sensor_df.loc[0]))
                
        dp.saveDataSet(arr, columns=columns, path=plc_csv_dir + '/new.csv')