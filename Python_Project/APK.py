
import pandas as pd
import matplotlib.pyplot as plt

# sample data set from apk
# NEP_mock_2014 = r'C:\Users\PETM_PC1\Documents\APK\APK\datagen\NEP_mock_2014.csv'
# NHBS_mock_2014 = r'C:\Users\PETM_PC1\Documents\APK\APK\datagen\NHBS_mock_2014.csv'
'''
NEP_ds = pd.read_csv(NEP_mock_2014)
NHBS_ds = pd.read_csv(NHBS_mock_2014)
print(NEP_ds[:20])
print(NHBS_ds[:20])
'''
#seriesplot = pd.DataFrame({'Age': NEP_ds['AGE'],'Zipcode' : NEP_ds['ZIPCODE']})
#seriesplot.cumsum(0).plot()
#NEP_ds.groupby('SEX').AGE.sum().plot(kind='bar')
#plt.show()

# Real time data
Population = r'C:\Users\PETM_PC1\Documents\Research docs\Dataset\exp_lhs1\exp_lhs1\instance_99\output\2016-03-08--10.35.14-1191944893\2016-03-08--10.35.14.populations.csv'
file = pd.read_csv(Population,skiprows=4)
print(file[:10])