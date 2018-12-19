
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

if __name__ == '__main__':
    dp = dataset_preprocess.DataSetPreprocess()
    
    dataset = "../datas/01-TrainingData-qLua/final_new.csv"
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    df = dp.loadDataSet(filename=dataset, columns=names)

    print(len(df))
    tmp = df[
        (df['spindle_load'] > 35.5) | (df['spindle_load'] < 3)
        | (df['y'] < -205)
        | (df['z'] > -100)
        | (df['vibration_1'] > 35.5) | (df['vibration_1'] < 28)
        | (df['vibration_2'] > 36) | (df['vibration_2'] < 28.7)
        | (df['vibration_3'] > 35.4) | (df['vibration_3'] < 28)
        | (df['current'] > 10)
    ]

    print(len(tmp.index))
    # for index in tmp.index:
    df = df.drop(index=list(tmp.index), axis=0)
    print(len(df))

    df = dp.abs(df)

    tmp = df[(df['z'] < 400)]

    print(len(tmp.index))
    # for index in tmp.index:
    df = df.drop(index=list(tmp.index), axis=0)

    df.to_csv("../datas/01-TrainingData-qLua/final_new1.csv", sep=',', index=False, columns=names)

    for name in names:
        gp = datasetGraphPlot.GraphPlot()
        y = list(df[name])

        print(df[name].describe())

        x = [i for i in range(0, len(y))]
        gp.plotScatter(x_lst=x, x_label='index', y_lst=y, y_label='current')
        gp.show(x_label=name, y_label='y')
    