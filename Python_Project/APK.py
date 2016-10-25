import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os
from datetime import timedelta
import matplotlib
import random
from matplotlib.collections import LineCollection

matplotlib.style.use('ggplot')
font = {'family': 'normal',
        'weight': 'bold',
        'size': 11}
plt.rc('font', **font)

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

try:
    def dataset_info(ds):
        # Info about dataset
        ds.info()
        ds.dtypes
        print(ds.head())
        print(ds.tail())
except Exception as r:
    print('exception encountered' + r)


def ignore_divByZero():
    # To ignore division by zero error
    np.seterr(divide='ignore')


def sub_plots():
    # create space between plots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    fig.subplots_adjust(hspace=1.0)


try:
    # Plotting statistics from a dataset
    def plot_stats(stats):
        # stats.hist()
        stats.boxplot(return_type='dict')
        plt.show()

except Exception as e:
    print(e.__cause__)

try:
    # Basic statistics of a Dataset
    def summary_statistics_dataset(dataset_name):
        print('\n')
        print('Summary statistics of a Dataset \n')
        result = dataset_name.describe()
        return result
except Exception as e:
    print("exception occurred : " + e)

try:
    # Basic statistics of a variable
    def summary_statistics(file_name, column_name):
        print('\n')
        print('Summary statistics of a variable \n')
        df = file_name[column_name].describe()
        print(df)
        print('\n')


    # Descriptive statistics- Mean/Average
    def statistics_Mean(file_name, column_name):
        print('statistics - mean \n')
        df = file_name[column_name].mean()
        print(df)
        print('\n')  # selecting columns, rows
        print(file_name['Tick'])
        print(file_name[['Tick', 'Year']])
        print(file_name[file_name.Year >= 2010])
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
        start_time = time.monotonic()
        print(path)
        df_pop = pd.DataFrame()
        df_population_params = []
        df_population = []
        df_events = []
        df_status = []
        df_statusregular = []
        files_directory = glob.glob(path)  # get all the csv files from the directory
        for file in files_directory:
            file_name = file_parsing(file)
            print("name of the file : " + file_name)
            if (file.__contains__("population")):
                df_params = pd.read_csv(file, skiprows=1, nrows=1)  # to extract params from populations file
                df = pd.read_csv(file, skiprows=4)  # to extract data from populations file
                df['file_name'] = file_name
                df_pop = df
                df_population.append(df)
                df_population_params.append(df_params)
            elif (file.__contains__("statusRegular")):
                df = pd.read_csv(file, skiprows=1)
                df_statusregular.append(df)
            elif (file.__contains__("status")):
                df = pd.read_csv(file, skiprows=1)
                df_status.append(df)
            elif (file.__contains__("events")):
                df = pd.read_csv(file, skiprows=1)
                df_events.append(df)
        if len(df_population) > 1:
            population_dataset = pd.concat(df_population, ignore_index=True)
            return population_dataset
        else:
            population_dataset = df_pop
            return population_dataset
        # statusRegular_dataset = pd.concat(df_statusregular)
        # status_dataset = pd.concat(df_status)
        # events_dataset = pd.concat(df_events)
        end_time = time.monotonic()
        print(timedelta(seconds=end_time - start_time))  # execution time of this code snippet

        # return population_dataset, statusRegular_dataset, status_dataset, events_dataset
except (FileNotFoundError, IndexError) as f:
    print('exception encountered' + f)

try:
    # Individual filename parsing
    def file_parsing(file):
        file_split = os.path.normpath(file)
        file_split = file_split.split('\\')
        name = file_split[6]
        instance = file_split[8]
        file_names = (name + '_' + instance + '_' + file_split[-1])
        print(file_names)
        return file_names
except Exception as e:
    print(e.__cause__)

try:
    # filename parsing
    def file_name_parsing(lhs):
        files_directory = glob.glob(lhs)
        file_names = []
        for file in files_directory:
            file_split = os.path.normpath(file)
            file_split = file_split.split('\\')
            name = file_split[6]
            instance = file_split[8]
            file_names.append(name + '_' + instance + '_' + file_split[-1])
        print(file_names)
        file_count = len(file_names)
        return file_names, file_count
except Exception as e:
    print(e.__cause__)

try:
    # Treatment average generator
    def average_ticks(datasets):
        groupby_ticks = datasets.groupby('Tick')
        lhs_avg = groupby_ticks.mean()  # mean
        lhs_std = groupby_ticks.std()  # standard deviation
        # print(lhs_avg)
        return lhs_avg
except Exception as e:
    print(e.__cause__)

try:
    # Plot prevalance for lhs based on ticks
    def plot_prevalance(dataset, column, dir):
        loc = r'C:/Users/PETM_PC1/Documents/Research_docs/Misc_docs'
        fig_location = loc + '/fig_' + column.__getitem__(1) + '_' + str(random.randint(0, 500)) + '_' + '.png'
        start_year = 2010
        days_per_year = 365
        y = start_year
        year_row = []
        cols_prevalence = []
        cols_HR = []
        for row in dataset['Year']:
            if row > 0:
                y = row
            year_row.append(int(y))
        dataset['Year'] = year_row  # Modifying the data in the 'year' column to reflect correct year's
        for cols in dataset.columns:
            if (cols.startswith('prevalence')):
                if not cols.startswith('prevalence_HCV') and not cols.startswith('prevalence_Syringe'):
                    cols_prevalence.append(cols)
                    # if (cols.endswith('=HR') and not cols.startswith('cured') and not cols.startswith('intreatment')):
                    #     cols_HR.append(cols)
        print(cols_prevalence)
        styles1 = ['bs-', 'ro-', 'y^-', 'rs-', 'go-', 'b^-', 'k^-', 'yD-', 'rv-', 'b+-', 'g*-', 'g1-']
        plt.legend(loc='best', fontsize=4)
        dataset.plot(x='Year', y='prevalence_Age=LEQ30', ylim=(0.0, 1.0), style=styles1,
                     title='Plot_Prevalance_LHS1')
        plt.savefig(fig_location)
except Exception as e:
    print(e.__cause__)

try:
    # Plot prevalance for lhs based on ticks for multiple columns
    def plot_prevalance_box(dataset, column, dir):
        loc = r'C:/Users/PETM_PC1/Documents/Research_docs/Misc_docs'
        fig_location = loc + '/fig_' + column + '.png'
        start_year = 2010
        days_per_year = 365
        y = start_year
        year_row = []
        for row in dataset['Year']:
            if row > 0:
                y = row
            year_row.append(int(y))
        dataset['Year'] = year_row  # Modifying the data in the 'year' column to reflect correct year's
        print(dataset.head())
        color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}
        fig, ax = plt.subplots()
        ax.plot('Year', column, linewidth=2)
        dataset.boxplot('RNApreval_ALL', xticks=range(2010, 2045, 1), ax=ax)
        plt.savefig(fig_location)
except Exception as e:
    print(e.__traceback__)


def dir_traverse(dir):
    for root, dirs, files in os.walk(dir, topdown=True):
        # for name in files:
        #     print(os.path.join(root, name))
        for name in dirs:
            print(os.path.join(root, name))
            dir = os.path.join(root, name)
    return dir


def plot_treatments(files):
    # prev = pd.DataFrame(columns=['RNApreval_ALL_LHS1', 'RNApreval_ALL_LHS2', 'RNApreval_ALL_LHS3', 'Year'])
    prev = pd.DataFrame(columns=['Unbiased Random Recruitment at 1%', 'Full Network Recruitment at 1%',
                                 'RNA prevalence without treatment', 'Year'])
    # prev = pd.DataFrame(
    # columns=['RNApreval_Race=NHBlack_LHS1', 'RNApreval_Race=NHBlack_LHS2', 'RNApreval_Race=NHBlack_LHS3', 'Year'])
    lhs1 = pd.read_csv(files.__getitem__(0))
    lhs2 = pd.read_csv(files.__getitem__(1))
    lhs3 = pd.read_csv(files.__getitem__(2))
    # print(lhs1.head(), lhs2.head(), lhs3.head())
    prev['Unbiased Random Recruitment at 1%'] = lhs1['RNApreval_ALL']
    prev['Full Network Recruitment at 1%'] = lhs2['RNApreval_ALL']
    prev['RNA prevalence without treatment'] = lhs3['RNApreval_ALL']
    prev['Year'] = lhs2.Year
    return prev


def plot_treatments_compare(files):
    prev = pd.DataFrame(
        columns=['RNApreval_ALL_With_Treatment', 'RNApreval_ALL_Without_Treatment'])
    file1 = pd.read_csv(files.__getitem__(0))
    file2 = pd.read_csv(files.__getitem__(1))
    print(file1.head(), file2.head())
    prev['RNApreval_ALL_With_Treatment'] = file1['RNApreval_ALL']
    prev['RNApreval_ALL_Without_Treatment'] = file2['RNApreval_ALL']
    prev['Year'] = file1.Year
    return prev


try:
    # Plot prevalance for lhs based on ticks
    def plot_prevalance_comparison(dataset, columns):
        loc = r'C:/Users/PETM_PC1/Documents/Research_docs/Misc_docs'
        fig_location = loc + '/fig_Prevalence_comparison' + '_' + str(random.randint(0, 500)) + '_' + '.pdf'
        start_year = 2015
        days_per_year = 365
        y = start_year
        year_row = []
        for row in dataset['Year']:
            if row > 0:
                y = row
            year_row.append(int(y))
        dataset['Year'] = year_row  # Modifying the data in the 'year' column to reflect correct year's
        print(dataset)
        dataset_new = dataset.groupby('Year').mean()
        print(dataset_new)
        styles1 = ['y-', 'b-', 'r-']
        width = [2, 2, 5]
        plt.legend(loc='best', fontsize=18, )
        plot1 = dataset_analysis.plot(x='Year', y=columns,
                              style=styles1, ylim=(0.0, 0.6))
        # plt.rcParams['axes.color'] = 'b'
        # plt.rcParams['grid.color'] = 'blue'
        lines = LineCollection(plot1, linewidths=width)
        fig, ax = plt.subplot()
        ax.add_collection(lines)
        plt.ylabel("HCV RNA Prevalence", {'size': '20'}, color='grey')
        plt.xlabel("Years of Treatment", {'size': '20'}, color='grey')
        # plt.title("Plot_RNApreval_All", {'size': '15'},color='grey')
        # plt.style.use(['dark_background'])
        plt.savefig(fig_location)
except Exception as e:
    print(e.__cause__)

if __name__ == '__main__':
    column_name = 'Year'

    try:
        lhs1 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs1/exp_lhs1/*/*/*/*.csv'  # Directory of LHS1 files
        lhs1_inst1 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs1/exp_lhs1/instance_8/*/*/*.csv'  # Directory of LHS1 files -instance 1
        lhs2 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs2/exp_lhs2/*/*/*/*.csv'  # Directory of LHS2 files
        lhs3 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs3/exp_lhs3/*/*/*/*.csv'  # Directory of LHS3 files
        lhs = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/'  # Directory of all LHS files

        # analysis files from various treatment methods
        pickle_lhs2 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs2/exp_lhs2/_analysis/dossier_summaries-2016_07_28__14_47_27.csv'
        pickle_lhs1 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs1/exp_lhs1/_analysis/dossier_summaries-2016_07_27__12_02_56.csv'
        pickle_lhs3 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs3/exp_lhs3/_analysis/dossier_summaries-2016_07_28__16_20_52.csv'

        # analysis files for comparison
        file1 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/exp_lhs1/exp_lhs1/_analysis/dossier_summaries-2016_07_27__12_02_56.csv'
        file2 = r'C:/Users/PETM_PC1/Documents/Research_docs/Dataset/APK_model_output/2016-07-25--04_00_32_populations_dossier.csv'

        pickles = []
        pickles.extend([pickle_lhs1, pickle_lhs2, file2])
        # pickles.extend([file1, file2])

        ## to plot treatment outputs
        dataset_analysis = plot_treatments(pickles)
        # dataset_analysis = plot_treatments_compare(pickles)

        columns = ['Unbiased Random Recruitment at 1%', 'Full Network Recruitment at 1%',
                   'RNA prevalence without treatment']
        # columns = ['RNApreval_ALL_With_Treatment', 'RNApreval_ALL_Without_Treatment']
        plot_prevalance_comparison(dataset_analysis, columns)

        # we can use os.path function to get the directory and identify treatment files

        # dir = os.listdir(lhs)
        # print(dir)

        # dir = dir_traverse(lhs)  # to traverse the entire directory
        # file_names, files_count = file_name_parsing(lhs)
        # print(file_names, files_count)



        datasets = fileHandling(lhs1)
        # print(datasets[0])
        # dataset_info(datasets)
        # Descriptive Stats
        # summary_statistics(column_name)
        # statistics_Mean(datasets, column_name)
        # summary_statistics_dataset(dataset_name)
        # correlation(file)
        # covariance(file)

        result = average_ticks(datasets)
        # plot the prevalence for a treatment scenario (LHS) based on ticks
        column = ['RNApreval_ALL', 'population_ALL']
        # plot_prevalance(result, column, lhs)
        plot_prevalance(result, column, lhs)

        # Summary statistics for LHS1 dataset
        # result_stats = summary_statistics_dataset(datasets)
        #  plotting the stats
        # plot_stats(result_stats)

    except Exception as e:
        print(e)
