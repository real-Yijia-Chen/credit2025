#### Import the necessary modules ####
import missingno as msno    # Import the missingno module
import numpy as np      # Import the numpy module
import os               # Import the os module    
import pandas as pd     # Import the pandas module
from pylab import *     # Import the pylab module
import random           # Import the sys and random module
import scikit_posthocs as sp # Import the posthocs module
import seaborn as sns   # Import the seaborn module 
import shutil           # Import the shutil module
import sys              # Import the sys module  
from linearmodels.panel import PanelOLS  # Import the PanelOLS module from the linearmodels.panel package
from matplotlib.font_manager import FontProperties  # Import the FontProperties module from the matplotlib.font_manager package
from matplotlib import pyplot as plt    # Import the pyplot module from the matplotlib package
from IPython.core.display import HTML   # Import the HTML module
from scipy import interpolate   # Import the interpolate module from the scipy package
from scipy.stats import friedmanchisquare     # Import the friedmanchisquare module from the scipy package
from scipy.stats import norm    # Import the norm module from the scipy package
from sklearn.preprocessing import StandardScaler  # Import the StandardScaler module from the sklearn package
from statsmodels.formula.api import ols  # Import the ols module from the statsmodels package
from stargazer.stargazer import Stargazer   # Import the Stargazer module 
from zipfile import ZipFile             # Import the zipfile module 


#### Set directory ####
dirname_main = 'your_dirname'
os.chdir(dirname_main)     # Use forward slash '/' in Windows or Linux
os.getcwd()             # Get the current working directory

#### Choose the measure of regional credit environment ####
credit_measure = '专家'   # The measure of regional credit environment. I chose expert/专家. Other two options are principal component/主成分 and equal weights/等权.

#### Font ####
plt.rcParams['font.sans-serif'] = ['SimSun']    # Set the font style
plt.rcParams['axes.unicode_minus'] = False      # Display the negative sign
plt.rcParams['figure.figsize'] = 12,8           # Set the figure size
plt.rcParams['legend.fontsize'] = 16            # Set the legend font style
plt.rcParams['legend.title_fontsize'] = 18      # Set the title font size 




##########################
### Unzip the raw data ###
##########################

# Directory name and file names
dirname_rawdata = dirname_main + '/Data/Raw'    # Directory name of the raw data
dirname_input = dirname_main + '/Data/Input'    # Directory name of the input data
filename_rawdata = '20240823  原始数据.zip'                                      # File name of the raw data
filename_suppdata = ['02.社会信用环境指数_原始数据0823 - 处理1.xlsx','03.社会信用环境指数_原始数据0823 - 综合得分-留.xlsx',
                     '社会信用环境指数_三级指标0824.xlsx','社会信用环境指数0824.xlsx','控制变量补充数据.xlsx']

filelist = os.listdir(dirname_input)    # List of files in the 'Input' directory
datalist = ['1.社会信用环境指数_原始数据0823.xlsx', '2.区域经济增长指标原始数据.xlsx', 
            '3.中介变量原始数据', '4.用于计算控制变量的初始数据.xlsx', 
            '02.社会信用环境指数_原始数据0823 - 处理1.xlsx','03.社会信用环境指数_原始数据0823 - 综合得分-留.xlsx',
            '社会信用环境指数_三级指标0824.xlsx','社会信用环境指数0824.xlsx',
            '社会信用环境评价指标体系构建0823.docx','控制变量补充数据.xlsx']    # List of files to be imported

if set(datalist) <= set(filelist):
    print('The datasets are already in the directory')
else:
    # Copy and paste the supplementary data to the 'Input' directory
    for filename in filename_suppdata:
        src = dirname_rawdata + '/' + filename
        dst = dirname_input + '/' + filename
        shutil.copyfile(src, dst)

    # Copy the raw data to the 'Input' directory
    src = dirname_rawdata + '/' + filename_rawdata  # Source file
    dst = dirname_input + '/' + filename_rawdata    # Destination file
    shutil.copyfile(src, dst)

    # Unzip the copied raw data
    with ZipFile(dst, 'r') as zObject: 
        # Extract all the members of the zip file in the 'Input' directory
        zObject.extractall(path = dirname_input) 

    # Erase the copied raw data
    os.remove(dst)




##################################################################
### Clean the credit score dataset: '社会信用环境指数0824.xlsx' ###
##################################################################

# Clear the screen
os.system('cls')    # For Windows

# Import the dataset
dataname = dirname_input + '/社会信用环境指数0824.xlsx'
datafile = pd.ExcelFile(dataname)  
print("The sheets in the dataset: ", datafile.sheet_names)  # Print the sheet names

with pd.ExcelFile(dataname) as data1:  
    df1 = pd.read_excel(data1, "Sheet1")   # Read the first sheet

# Print the basic information of the dataset
print(f" \n Number of observations: {len(df1)} \n ")   # Print the number of variables
print(f" \n Data types: {df1.dtypes} \n ")             # Print the data types

# Rename the columns '省' and '市' to 'province' and 'city'
df1.rename(columns={'省':'province','市':'city'}, inplace=True)    # Rename the '省' column to 'province' and '市' column to 'city'

# Check whether the dataset has a unique key
print(df1['id'].is_unique)    # Check whether the 'id' variable is unique, the result is False

# Check the number of missing values in each variable
print(f"\n Number of missing values in each variable: {df1.isna().sum().sort_values()} \n")   # No missing values for all variables

# Print the number of unique values of year, provinces, and cities
for var in ['year','province','city']:
    print(f"\n Number of unique values of {var}: {df1[var].nunique()} \n")    # 15 years, 31 provinces, and 287 cities

# Print the unique values of year, provinces, and cities
for var in ['year','province','city']:
    print(f"\n Number of unique values of {var}: {df1[var].unique()} \n")    # 15 years, 31 provinces, and 287 cities

# Remove any spaces in the 'city' column
df1['city'] = df1['city'].str.strip()    # Remove any spaces in the 'city' column

# Remove the character '市' in the 'city' column
df1['city'] = df1['city'].str.replace('市','')    # Remove the character '市' in the 'city' column

# For each year, print the number of observations
print(f"\n Number of observations in each year: \n {df1['year'].value_counts()} \n")  # Each year has exactly 287 observations, balanced panel data

# Split the datasets into three subsets conditional on the measures of the credit environment
print(df1.columns)                  # Print the column names
varlist_common = df1.columns[0:4]   # Column names in common: 'id', 'year', 'province', 'city'
stringlist = ['专家','等权','主成分']  # Strings to be matched
namelist = stringlist  # Names of the subsets
dict_temp = {}  
for i in range(3):
    # Choose the variables with string '专家', '等权', and '主成分'
    dict_temp[namelist[i]] = df1[varlist_common.append(df1.columns[df1.columns.str.contains(stringlist[i])])]  

# Check the distribution of each variable in each chunk
varlist_chunk = ['社会信用环境指数','制度','技术','应用','教育|载体|文化']  # Chunks to be checked: credit scores, institutions, technology, application, education|carriers|culture
for key in dict_temp.keys():
    # Create variable lists
    varlist = {}    # Create an empty dictionary
    for i in range(len(varlist_chunk)):
        varlist[i] = dict_temp[key].columns.str.contains(varlist_chunk[i])
        print(f"\n \n \n Summary statistics of variables in {key} with string '{varlist_chunk[i]}': \n")
        print(dict_temp[key].loc[:,dict_temp[key].columns[varlist[i]]].describe())  # Print the summary statistics of each chunk of variables

# The result suggests that the principal component measure of credit environment yields credit scores ranging from -16 to 847.
# Other two measures yield credit scores ranging from 0 to 100, with much smaller variances.

# Save the subsets to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/credit_score.xlsx') as writer:  # Create an ExcelWriter object
    for i in range(3):
        # Sort each dataset by id and year
        dict_temp[namelist[i]].sort_values(by=['id','year'], inplace=True)

        # Save the subsets with string '专家', '等权', and '主成分'
        dict_temp[namelist[i]].to_excel(writer, sheet_name = namelist[i], index=False)   




#####################################################################################
### Clean the regional characteristics dataset: '4.用于计算控制变量的初始数据.xlsx' ###
#####################################################################################

# Clear the screen
os.system('cls')    # For Windows

# Import the dataset
dataname = dirname_input + '/4.用于计算控制变量的初始数据.xlsx'
datafile = pd.ExcelFile(dataname)  
print("The sheets in the dataset: ", datafile.sheet_names)  # Print the sheet names

with pd.ExcelFile(dataname) as data2:  
    df2 = pd.read_excel(data2, "Sheet1")   # Read the first sheet

# Print the basic information of the dataset
print(f" \n Number of observations: {len(df2)} \n ")   # Print the number of variables
print(f" \n Data types: {df2.dtypes} \n ")             # Print the data types

# Rename the columns '年份', '行政区划代码' and '城市' to 'year', 'id' and 'city'
df2.rename(columns={'年份':'year','行政区划代码':'id','城市':'city'}, inplace=True)    # Rename the '行政区划代码' column to 'id' and '城市' column to 'city'

# Remove any spaces in the 'city' column
df2['city'] = df2['city'].str.strip()    # Remove any spaces in the 'city' column

# Check the number of missing values in each variable
print(f"\n Number of missing values in each variable: {df2.isna().sum().sort_values()} \n")   # There are some missing values
msno.matrix(df2)    # Visualize the missing values
plt.show()          # There are many missing values in certain years for certain cities

# Print the number of unique values of year and cities
for var in ['year','city']:
    print(f"\n Number of unique values of {var}: {df2[var].nunique()} \n")    # 15 years, 300 cities

# Print the unique values of year and cities
for var in ['year','city']:
    print(f"\n Number of unique values of {var}: {df2[var].unique()} \n")    # 15 years, 300 cities

# Since the regional characteristic dataset includes cities that are missing in the credit score dataset, we need to drop these cities
citylist = df1['city'].unique()    # Unique city values in the credit score dataset
df2 = df2[df2['city'].isin(citylist)]   # Drop the observations with cities that are not in the credit score dataset
print(df2['city'].nunique())    # Now only 287 cities remain

# Sort the dataset by id and year
df2.sort_values(by=['id','year'], inplace=True)
msno.matrix(df2)    # Visualize the missing values again
plt.show()          # Some cites contribute to most of the missing values

# For each year, print the number of observations
print(f"\n Number of observations in each year: \n {df2['year'].value_counts()} \n")  # Each year has exactly 287 observations, balanced panel data

# Check the distribution of each variable by year 
yearlist = df2['year'].unique()     # Unique year values
varlist1 = df2.columns[3:8]         # Variables of interest
varlist2 = df2.columns[8:]          # Variables of interest
for n in yearlist:
    print(f"\n \n \n Summary statistics of variables in year '{n}': \n")
    
    # Print the summary statistics of each variables of interest by year
    print(df2[df2['year']==n].loc[:,varlist1].describe()) 
    print(df2[df2['year']==n].loc[:,varlist2].describe()) 

# So far the distribution of each variable seems to be normal

#################### Identify missing values in the regional characteristic dataset ####################

# Print the missing values by variable, city, and year
missing_list = []                   # Create an empty list
idx_count = -1                      # Initialize the index counter
varlist = df2.columns[3:]           # Variables of interest
headerlist = list(df2.columns)      # Column names
print(varlist)

# Iterate over the rows of the dataset and check if there are any missing values
for idx, row in df2.iterrows():
    if row.isna().sum() > 0:  # Check if the row contains any missing values
        idx_count += 1
        missing_row = []
        missing_row.append(df2['year'][idx])
        missing_row.append(df2['id'][idx])
        missing_row.append(df2['city'][idx])
        
        # Check each variable in varlist for missing values
        for var in varlist:
            if pd.isna(row[var]):  # Check if the specific variable in the row is NaN
                missing_row.append(1)
            else:
                missing_row.append(0)

        # Append the list 'missing_row' to the list of list 'missing_list':
        missing_list.append(missing_row)

# Convert 'missing_list' to a dataframe
df_missing = pd.DataFrame(missing_list, columns=headerlist)  # Convert the list of list to a dataframe

# Save the missing value dataframe to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/regional_characteristic_missing_values.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_missing.to_excel(writer, sheet_name = '区域特征', index=False)    # Save the dataset

#################### Import the supplementary dataset '控制变量补充数据.xlsx' ####################

# Import the supplementary dataset
with pd.ExcelFile(dirname_input + '/控制变量补充数据.xlsx') as data_supp:  
    df_supp = pd.read_excel(data_supp, "区域特征", nrows=60)   # Read the first sheet

# Fill the missing values for Beijing, Tianjin, Shanghai, and Chongqing with supplementary data
supplist = ['北京','天津','上海','重庆']
for var in supplist:
    # Copy the supplementary data to the dataframe 'df2'
    df2.loc[df2['city']==var,['城区面积平方公里','城区人口万人']] = df_supp.loc[df_supp['city']==var,['城区面积平方公里','城区人口万人']].values

    # Fill the missing area by backward filling, since the city area appears to be very stable over time
    df2.loc[df2['city']==var, '城区面积平方公里'] = df2.loc[df2['city']==var, '城区面积平方公里'].bfill() 

    # Then, fill the missing area by fowrad filling.
    df2.loc[df2['city']==var, '城区面积平方公里'] = df2.loc[df2['city']==var, '城区面积平方公里'].ffill() 

# Fill the missing values for Bijie, Tongren and Haidong by interpolation
interplist = ['毕节','铜仁','海东']
for var in interplist:
    if (var == '毕节') | (var == '铜仁'):
        collist = ['城区面积平方公里','城区人口万人']
    else:
        collist = varlist[0:6]
    for col in collist:
        if col=='城区面积平方公里': # Fill the missing area by backward filling, since the city area appears to be very stable over time
            df2.loc[df2['city']==var, col] = df2.loc[df2['city']==var, col].bfill()
            print(df2.loc[df2['city'] == var, col])
        else:   # Fill the rest of missing values by linear extrapolation
            date_series = df2[df2['city']==var]['year']
            target_series = df2[df2['city']==var][col]
            idx = df2[df2['city']==var][col].notna()
            x = date_series[idx].values
            log_y = np.log(target_series[idx].values)   # Take the log of the target variable to ensure the predicted values are positive
            f = interpolate.interp1d(x, log_y, kind='linear', fill_value='extrapolate')
            x_new = date_series[~idx].values
            y_new = np.exp(f(x_new))
            df2.loc[(df2['city'] == var) & (~idx), col] = y_new
            print(df2.loc[(df2['city'] == var) & (~idx), col])

print(df2[df2['city']=='海东'])

# Rename the year column and save the dataset to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/regional_characteristics.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df2.to_excel(writer, sheet_name = '区域特征', index=False)    # Save the dataset




###########################################################################
### Clean the regional economy dataset: '2.区域经济增长指标原始数据.xlsx' ###
###########################################################################

# Clear the screen
os.system('cls')    # For Windows

# Import the dataset
dataname = dirname_input + '/2.区域经济增长指标原始数据.xlsx'
datafile = pd.ExcelFile(dataname)  
print("The sheets in the dataset: ", datafile.sheet_names)  # Print the sheet names

with pd.ExcelFile(dataname) as data3:  
    df3 = pd.read_excel(data3, "Sheet1")   # Read the first sheet

# Print the basic information of the dataset
print(f" \n Number of observations: {len(df3)} \n ")   # Print the number of variables
print(f" \n Data types: {df3.dtypes} \n ")             # Print the data types

# Rename the columns '年份', '行政区划代码' and '地区' to 'year', 'id' and 'city'
df3.rename(columns={'年份':'year','行政区划代码':'id','地区':'city'}, inplace=True)    # Rename the '行政区划代码' column to 'id' and '城市' column to 'city'

# Remove any spaces in the 'city' column
df3['city'] = df3['city'].str.strip()    # Remove any spaces in the 'city' column

# Sort the dataset by id and year, and then display the number of missing values in each variable
df3.sort_values(by=['id','year'], inplace=True)
msno.matrix(df3)    # Visualize the missing values again
plt.show()          # Some cites contribute to most of the missing values

# Print the number of unique values of year and cities
for var in ['year','city']:
    print(f"\n Number of unique values of {var}: {df3[var].nunique()} \n")    # 17 years, 300 cities

# Print the unique values of year and cities
for var in ['year','city']:
    print(f"\n Number of unique values of {var}: {df3[var].unique()} \n")    # 17 years, 300 cities

# Since the regional economy dataset includes cities and years that are missing in the credit score dataset, we need to drop some observations
citylist = df1['city'].unique()    # Unique city values in the credit score dataset
yearlist = df1['year'].unique()    # Unique year values in the credit score dataset
df3 = df3[(df3['city'].isin(citylist)) & (df3['year'].isin(yearlist))]   # Drop the observations with cities that are not in the credit score dataset
print(df3['year'].nunique())    # Now only 15 years remain
print(df2['city'].nunique())    # Now only 287 cities remain

# Check the distribution of missing values again
df3.sort_values(by=['id','year'], inplace=True)
msno.matrix(df3)    # Visualize the missing values again
plt.show()          # Only the 'fixed asset investment' variable has missing values

# For each year, print the number of observations
print(f"\n Number of observations in each year: \n {df2['year'].value_counts()} \n")  # Each year has exactly 287 observations, balanced panel data

# Check the distribution of each variable by year 
yearlist = df3['year'].unique()     # Unique year values
varlist1 = df3.columns[3:7]         # Variables of interest
varlist2 = df3.columns[7:]          # Variables of interest
for n in yearlist:
    print(f"\n \n \n Summary statistics of variables in year '{n}': \n")
    
    # Print the summary statistics of each variables of interest by year
    print(df3[df3['year']==n].loc[:,varlist1].describe()) 
    print(df3[df3['year']==n].loc[:,varlist2].describe()) 

# There are three anomalies:
# 1. Some areas have negative GDP per capita in 2008 and 2009
# 2. Some areas have zero fixed asset investment price index in 2008-2022
# 3. Some areas have negative fixed asset investment in 2022 (This could happen if asset depreciation exceeds new investment)

#################### Check the GDP per capita ####################

print(df3[df3['人均地区生产总值元'] <= 0])     # Haidong/海东 has negative GDP per capita in 2008 and 2009

# Replace negative GDP per capita by linear extrapolation
date_series = df3[df3['city']=='海东']['year']
target_series = df3[df3['city']=='海东']['人均地区生产总值元']
idx = (df3['city']=='海东') & (df3['人均地区生产总值元'] > 0)
x = date_series[idx].values
log_y = np.log(target_series[idx].values)   # Log-transform the dependent variables to ensure positive predicted values
f = interpolate.interp1d(x, log_y, kind='linear', fill_value='extrapolate')
x_new = date_series[~idx].values
y_new = np.exp(f(x_new))
df3.loc[(df3['city']=='海东') & (df3['人均地区生产总值元'] <= 0),'人均地区生产总值元'] = y_new
print(df3[df3['city']=='海东'])

#################### Check fixed asset investment price index ####################

print(df3[df3['FPI_固定资产投资价格指数']<=0.9])

# It turns out that Lhasa/拉萨 has zero fixed asset investment price index, and its fixed investments are all missing. 
# Replace Lhasa's fixed asset investment price index by NaN.
df3.loc[df3['city']=='拉萨','FPI_固定资产投资价格指数'] = df3.loc[df3['city']=='拉萨','FPI_固定资产投资价格指数'].replace(0, np.nan, inplace=True)
print(df3.loc[df3['city']=='拉萨','FPI_固定资产投资价格指数'])

#################### Check fixed asset investment ####################

print(df3[df3['固定资产投资亿元']<=10])

# The fixed asset investment of Baotou/包头 in year 2017-2022 is extremely small. 
# The fixed asset investment of Fujian cities in 2021 is extremely small as well.
# All cities with negative fixed asset investment are within Fujian/福建 province, and the year was 2022. 
# Modify the data by linear extrapolation

citylist = df3[df3['固定资产投资亿元']<=10]['city'].unique()
yearlist = df3['year'].unique()
print(citylist)
print(yearlist)
for var in citylist:
    date_series = df3[df3['city']==var]['year']
    target_series = df3[df3['city']==var]['固定资产投资亿元']
    idx = (df3['city']==var) & (df3['固定资产投资亿元']>10)
    x = date_series[idx].values
    log_y = np.log(target_series[idx].values) 
    f = interpolate.interp1d(x, log_y, kind='linear', fill_value='extrapolate')
    x_new = date_series[~idx].values
    y_new = np.exp(f(x_new))
    df3.loc[(df3['city']==var) & (df3['固定资产投资亿元']<=10),'固定资产投资亿元'] = y_new

#################### Identify missing values in the regional economy dataset ####################

# Print the missing values by variable, city, and year
missing_list = []                   # Create an empty list
idx_count = -1                      # Initialize the index counter
varlist = df3.columns[3:]           # Variables of interest
headerlist = list(df3.columns)      # Column names
print(varlist)

# Iterate over the rows of the dataset and check if there are any missing values
for idx, row in df3.iterrows():
    if row.isna().sum() > 0:  # Check if the row contains any missing values
        idx_count += 1
        missing_row = []
        missing_row.append(df3['year'][idx])
        missing_row.append(df3['id'][idx])
        missing_row.append(df3['city'][idx])
        
        # Check each variable in varlist for missing values
        for var in varlist:
            if pd.isna(row[var]):  # Check if the specific variable in the row is NaN
                missing_row.append(1)
            else:
                missing_row.append(0)

        # Append the list 'missing_row' to the list of list 'missing_list':
        missing_list.append(missing_row)

# Convert 'missing_list' to a dataframe
df_missing = pd.DataFrame(missing_list, columns=headerlist)  # Convert the list of list to a dataframe

# Save the missing value dataframe to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/regional_economy_missing_values.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_missing.to_excel(writer, sheet_name = '区域经济增长', index=False)    # Save the dataset

# The fixed asset investment data of four cities, Bijie/毕节, Tongren/铜仁, Lhasa/拉萨, and Haidong/海东, are missing.
# Also, the fixed assest investment price index of Lhasa/拉萨, is missing as well.

# Rename the year column and save the dataset to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/regional_economy.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df3.to_excel(writer, sheet_name = '区域经济', index=False)    # Save the dataset




##############################################################################################
### Merge the credit score, the regional characteristic, and the regional economy datasets ###
##############################################################################################

# Import the cleaned datasets
filename = dirname_output + '/credit_score.xlsx'
with pd.ExcelFile(filename) as data1:  
    df1 = pd.read_excel(data1, credit_measure)   # Read the principal component sheet
filename = dirname_output + '/regional_characteristics.xlsx'
with pd.ExcelFile(filename) as data2:  
    df2 = pd.read_excel(data2, "区域特征")   # Read the regional characteristics sheet
filename = dirname_output + '/regional_economy.xlsx'
with pd.ExcelFile(filename) as data3:  
    df3 = pd.read_excel(data3, "区域经济")   # Read the regional economy sheet

# Left-join the three datasets. Keep all observations in the credit environment dataset.
df_merged = df1.merge(df2, on=['year','city'], suffixes=('_信用库','_区域库'), how='left', validate='one_to_one')\
    .merge(df3, on=['year','city'], suffixes=('','_经济库'), how='left', validate='one_to_one') # Merge the three datasets with two keys: year and city
print(df_merged.columns)    # Print the column names of the merged dataset

# The merged dataset includes three duplicated id columns, two fixed asset investment columns, and two GDP columns.

#################### Double-check and modify duplicated columns ####################

# Drop duplicated id columns
df_merged = df_merged.drop(['id','id_区域库'], axis=1)
df_merged = df_merged.rename(columns={'id_信用库':'id'})
print(df_merged.columns)

# Compare the two fixed asset investment columns
df_fa = df_merged.loc[:,['year','city','固定资产投资亿元','固定资产投资亿元_经济库']]
print(df_fa.head(20))
df_fa['difference'] = df_fa['固定资产投资亿元'] - df_fa['固定资产投资亿元_经济库']  # Compute the difference

# If the absolute difference between the two fixed investment statistics is greater than 10^(-6), then the corresponding observation is identified as an anomaly.
df_fa_anomaly = df_fa.loc[np.absolute(df_fa['difference']) > 10**(-6),:]   

# Examine the anomalies
print(len(df_fa_anomaly))   # There are in total 84 fixed asset investment anomalies
print(df_fa_anomaly.head(50))

# Save the anomaly dataframe to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/fixed_asset_investment_anomaly.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_fa_anomaly.to_excel(writer, sheet_name = credit_measure+'_固定资产投资', index=False)    # Save the dataset

# By comparing the anomalies, it turns out that the fixed asset investment statistics in the 'regional economy' dataset should be the correct one.
df_merged.drop('固定资产投资亿元', axis=1, inplace=True)
df_merged.rename(columns={'固定资产投资亿元_经济库':'固定资产投资亿元'}, inplace=True)

# Compare the two GDP columns
df_GDP = df_merged.loc[:,['year','city','地区生产总值万元','地区生产总值万元_经济库']]
print(df_GDP.head(20))
df_GDP['difference'] = df_GDP['地区生产总值万元'] - df_GDP['地区生产总值万元_经济库']  # Compute the difference

# If the absolute difference between the two GDP statistics is greater than 10^(-6), then the corresponding observation is identified as an anomaly.
df_GDP_anomaly = df_GDP.loc[np.absolute(df_GDP['difference']) > 10**(-6),:]   

# Examine the anomalies
print(len(df_GDP_anomaly))   # There are in total 80 GDP anomalies
print(df_GDP_anomaly.head(50))

# Save the anomaly dataframe to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/GDP_anomaly.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_GDP_anomaly.to_excel(writer, sheet_name = credit_measure + '_固定资产投资', index=False)    # Save the dataset

# It turns out that the GDP statistics in the 'regional economy' dataset should be the correct one.
df_merged.drop('地区生产总值万元', axis=1, inplace=True)
df_merged.rename(columns={'地区生产总值万元_经济库':'地区生产总值万元'}, inplace=True)

#################### Clean the merged dataset ####################

# Rename all variables with the subscript '主成分'/PCA
subscript = '_' + credit_measure
varlist = df_merged.columns[df_merged.columns.str.contains(subscript)]
print(varlist)
for var in varlist:
    df_merged.rename(columns={var:var.replace(subscript,'')}, inplace=True)
print(df_merged.columns)

# Count the number of missing values in the merged dataset
print(df_merged.isna().sum().sort_values())     # At most 60 missing values in a variable

# Sort the dataset by id and year, and then display the number of missing values in each variable.
df_merged.sort_values(by=['id','year'], inplace=True)
msno.matrix(df_merged)    # Visualize the missing values again
plt.show()          # Some cites contribute to most of the missing values

# Save the merged dataset to the 'Output' directory
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
with pd.ExcelWriter(dirname_output + '/merged_data.xlsx') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_merged.to_excel(writer, sheet_name = credit_measure+'_合并数据', index=False)    # Save the dataset





