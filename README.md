#官方：
- 根据CPS框架收集控制器(PLC)信号和外置传感器(Sensor)信号，收集加工过程中的工况信息和传感器数据，以实现刀具磨耗在线监测与寿命预测为目标，建立诊断模型判定刀具磨耗状态。
    本次初赛共提供9组CNC加工数据，数据来源为实际CNC加工过程中，一把全新的刀具开始进行正常加工程序，直到刀具寿命终止时停止数据采集。
    并将这9组数据，分成4组训练数据(training data)以及5组测试数据(testing data)。
    4组训练数据将提供加工过程的PLC信号以及震动传感器的原始信号值，同时提供当组实验数据的完整加工「刀具全寿命时间」，单位为时间(minutes)。
    震动传感器的原始信号值因数据量极大，仅提供每5分钟任取1分钟的片段数据做为训练样本，并依据时间顺序1.csv, 2.csv…. n.csv提供，其中最后一个csv档即为实验结束前最终5分钟的撷取数据内容。
    在数据采样频率方面，PLC信号采样频率为33Hz（0.0303s/次，30.3ms/次），震动传感器采样频率25600Hz（0.000039s/次，0.039ms/次）。
    请参赛选手依据这些信息，预测测试刀具的剩余寿命，单位为时间(minutes)。

- 比赛链接地址：http://www.industrial-bigdata.com/datacompetition
# 数据集说明：    
## 训练集：  
- PLC参数：  
        time：记录时间  
        spindle_load：主轴负载  
        x：x轴机械坐标  
        y：y轴机械坐标  
        z：z轴机械坐标  
        csv_no：对应的sensor csv文件  
- sensor参数：  
        vibration_1：x轴方向振动信号  
        vibration_2：y轴方向振动信号  
        vibration_3：z轴方向振动信号  
        current：第一相电流信号  

# 步骤：
## 流程  
    分析数据—>数据合并->数据预处理->拟合曲线观察趋势->算法选型->预测结果->测试验证->计算准确率
### 分析数据：
    1）数据相互关系：
        a.在数据采样频率方面，PLC信号采样频率为33Hz（0.0303s/次，30.3ms/次），震动传感器采样频率25600Hz（0.000039s/次，0.039ms/次）。
        30.3 / 0.039 = 777;
        或者
        按照plc中csv_no为1的行数，与对应sensor的1.csv中的行数做倍数，根据每个sensor文件的行数不同而动态变化；
    b.振动传感器数据：
        由于振动传感器的数据，是使用1分钟的数据代替5分钟的数据；
        因此，按照倍数进行截取，并取平均值来代表这一笔数据；
    2）数值符号：
        x,y,z,vibration_1,vibration_2,vibration_3看作矢量；
        先去矢量化，再按列取平均值；
### 数据合并：  
    以第一把刀具为例：
    接口（1）：		函数名：samplingSensorData()
        对plc下的所有sensor数据，按照csv_no字段的值，对对应的sensor.csv文件做如下操作：
        1）计算sensor与plc对应csv_no数据的倍数关系 n；（考虑是否需要这样优化？）
        2）每行plc数据，对应 n 条sensor数据，不足 n 的，以实际剩余长度对应到plc对应csv_no的最后一笔数据；
        3）将截取的 n 条数据去矢量化；
        4）将去矢量化后的数据，按列取平均值，得到一行dataframe；
    接口（2）：		函数名：contact(sensor_df)
        1）将接口（1）中的结果，按列扩展对应plc的那一行数据；
        2）打标签：last_time，表示刀具剩余的时间，采用 “刀具全寿命 - csv_no*5”作为label值：
                    例如：    240 - 1*5 = 235
        3）返回拼接后的结果（应该是一行dataframe），给调用者做整合；
        
    调用者顺序：
        1）创建空的dataframe，columns为整合之后的字段：
                ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
                df_new = Dataframe(columns=['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time'])
        2）调用顺序：
                sensor_df_sampled = samplingSensorData()
                plc_df = contact(sensor_df)
                df_new.append(plc_df)

### 数据预处理：
### 拟合曲线观察趋势：
### 算法选型：
### 预测结果：
### 测试验证：
### 计算准确率：

# 程序目录说明

    bin：正式执行主入口
    datas：可放入数据集（也可以在程序中指定任何可访问的路径）
    docs：相关技术文档
    src：
        samplingSensorData.py：处理sensor数据，截取、去矢量化、取平均的类
        dataSetStatisticAnalysis：数据预处理和数据可视化的类
    tests：
        dataset_test.py：不取平均值的数据合并测试
        test.py：使用取平均值的方式的数据合并测试（测试时，只需要修改test()函数中的 plc_csv_dir 和filename变量即可）