import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# sample data set from apk
# NEP_mock_2014 = r'C:\Users\PETM_PC1\Documents\APK\APK\datagen\NEP_mock_2014.csv'
# NHBS_mock_2014 = r'C:\Users\PETM_PC1\Documents\APK\APK\datagen\NHBS_mock_2014.csv'
'''
NEP_ds = pd.read_csv(NEP_mock_2014)
NHBS_ds = pd.read_csv(NHBS_mock_2014)
print(NEP_ds[:20])
print(NHBS_ds[:20])
'''
# seriesplot = pd.DataFrame({'Age': NEP_ds['AGE'],'Zipcode' : NEP_ds['ZIPCODE']})
# seriesplot.cumsum(0).plot()
# NEP_ds.groupby('SEX').AGE.sum().plot(kind='bar')
# plt.show()

# Real time data
Population = r'C:\Users\PETM_PC1\Documents\Research docs\Dataset\exp_lhs1\exp_lhs1\instance_99\output\2016-03-08--10.35.14-1191944893\2016-03-08--10.35.14.populations.csv'
file = pd.read_csv(Population, skiprows=4)


def dataset_info():
    # Info about dataset
    file.info()
    file.dtypes
    print(file.head())
    print(file.tail())


# filter data
numeric_data = file.drop(['prevalence_HCV=cured', 'RNApreval_HCV=cured',
                          'prevalence_HCV=unknown', 'RNApreval_HCV=unknown',
                          'prevalence_HCV=ABPOS', 'RNApreval_HCV=ABPOS'], 1)


def ignore_divByZero():
    # To ignore division by zero error
    np.seterr(divide='ignore')


def selection_by_location():
    # selection by location
    # use loc for label based indexing and iloc for position based indexing
    print("selection by location ", file.iloc[[1, 50, 100]])


def sub_plots():
    # create space between plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    fig.subplots_adjust(hspace=1.0)


try:
    # Basic statistics of a Dataset
    def summary_statistics_dataset(dataset_name):
        print('\n')
        print('Summary statistics of a Dataset \n')
        for ds in dataset_name:
            print(ds.describe())
        print('\n')


    # Basic statistics of a variable
    def summary_statistics(column_name):
        print('\n')
        print('Summary statistics of a variable \n')
        df = numeric_data[column_name].describe()
        print(df)
        print('\n')


    # Descriptive statistics- Mean/Average
    def statistics_Mean(column_name):
        print('statistics - mean \n')
        df = numeric_data[column_name].mean()
        print(df)
        print('\n')  # selecting columns, rows
        print(numeric_data['Tick'])
        print(file[['Tick', 'Year']])
        print(file[file.Year >= 2010])
except RuntimeWarning as obj:
    print(obj.__cause__)

try:
    def correlation(Matrix):
        result = Matrix.corr()
        print(result)
except RuntimeError as r:
    print('exception encountered' + r)

try:
    def covariance(Matrix):
        result = Matrix.cov()
        print(result)
except RuntimeError as r:
    print('exception encountered' + r)

try:
    def fileHandling(path):
        print(path)
        listOfFiles = []
        df_population = []
        df_events = []
        df_status = []
        df_statusregular = []
        df = []
        files_directory = glob.glob(path)  # get all the csv files from the directory
        for file in files_directory:
            print("name of the file : " + file)
            if (file.__contains__("population")):
                df = pd.read_csv(file, skiprows=4)
                df_population.append(df)
            elif (file.__contains__("statusRegular")):
                df = pd.read_csv(file, skiprows=1)
                df_statusregular.append(df)
            elif (file.__contains__("status")):
                df = pd.read_csv(file, skiprows=1)
                df_status.append(df)
            elif (file.__contains__("events")):
                df = pd.read_csv(file, skiprows=1)
                df_events.append(df)
                # listOfFiles.append(df)
        population_dataset = pd.concat(df_population)
        statusRegular_dataset = pd.concat(df_statusregular)
        status_dataset = pd.concat(df_status)
        events_dataset = pd.concat(df_events)
        # print(population_dataset)
        return population_dataset, statusRegular_dataset, status_dataset, events_dataset
except FileNotFoundError as f:
    print('exception encountered' + f)

if __name__ == '__main__':
    column_name = 'Year'
    dataset_name = file

    # dataset_info()
    # Descriptive Stats
    # summary_statistics(column_name)
    # statistics_Mean(column_name)
    # summary_statistics_dataset(dataset_name)
    # correlation(file)
    # covariance(file)

    lhs1 = r'C:/Users/PETM_PC1/Documents/Research docs/Dataset/exp_lhs1/exp_lhs1/*/*/*/*.csv'  # Directory of LHS1 files
    datasets = fileHandling(lhs1)
    print(datasets[0])
    # Summary statistics for LHS1 dataset
    summary_statistics_dataset(datasets)

