
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

if __name__ == '__main__':
    dp = dataset_preprocess.DataSetPreprocess()
    
    dataset = "../../01-TrainingData-qLua/final.csv"
    names = ['spindle_load', 'x', 'y', 'z', 'vibration_1', 'vibration_2', 'vibration_3', 'current', 'last_time']
    df = dp.loadDataSet(filename=dataset, columns=names)

    df = dp.abs(df)

    print(len(df))
    tmp = df[
        (df['spindle_load'] > 35.5) | (df['spindle_load'] < 3)
        | (df['z'] < 400)
        | (df['vibration_1'] > 35.5) | (df['vibration_1'] < 28)
        | (df['vibration_2'] > 36) | (df['vibration_2'] < 28.7)
        | (df['vibration_3'] > 35) | (df['vibration_3'] < 28)
        | (df['current'] > 15)
    ]

    print(len(tmp.index))
    # for index in tmp.index:
    # df = df.drop(index=list(tmp.index), axis=0)
    print(len(df))

    # for index in tmp.index:
    df = df.drop(index=list(tmp.index), axis=0)

    # df.to_csv("../../01-TrainingData-qLua/final_new1.csv", sep=',', index=False, columns=names)

    for name in names:
        # if name != 'z':
        #     continue
        gp = datasetGraphPlot.GraphPlot()
        y = list(df[name])

        print(df[name].describe())

        x = [i for i in range(0, len(y))]
        gp.plotScatter(x_lst=x, x_label='index', y_lst=y, y_label='current')
        gp.show(x_label=name, y_label='y')
    