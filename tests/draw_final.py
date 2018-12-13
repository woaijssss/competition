
import src.datasetStatisticAnalysis.dataset_preprocess as dataset_preprocess
import src.datasetStatisticAnalysis.datasetGraphPlot as datasetGraphPlot

if __name__ == '__main__':
    gp = datasetGraphPlot.GraphPlot()
    dp = dataset_preprocess.DataSetPreprocess()
    
    dataset = "../../01-TrainingData-qLua/final.csv"
    names = ['spindle_load', 'x', 'y', 'z', 'last_time', 'vibration_1', 'vibration_2', 'vibration_3', 'current']
    df = dp.loadDataSet(filename=dataset, columns=names)
    
    current_lst_y = list(df['current'])
    
    x = [i for i in range(0, len(current_lst_y))]
    gp.plotScatter(x_lst=x, x_label='index', y_lst=current_lst_y, y_label='current')
    gp.show(x_label='current trend', y_label='current')
    