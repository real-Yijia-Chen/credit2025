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
from cycler import cycler   # Import the cycler module from the cycler package
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
os.chdir('D:国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy')     # Use forward slash '/' in Windows or Linux
os.getcwd()             # Get the current working directory

#### Choose the measure of regional credit environment ####
credit_measure = '专家'   # The measure of regional credit environment. I chose expert/专家. Other two options are principal component/主成分 and equal weights/等权.

#### Font ####
plt.rcParams['font.sans-serif'] = ['SimSun']    # Set the font style
plt.rcParams['axes.unicode_minus'] = False      # Display the negative sign




#################################################################################################
### Analyze the (non)linear relationship between regional GDP and regional credit environment ###
#################################################################################################

# Set directories
dirname_output = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Data/Output'    # Directory name of the output data
dirname_result = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Results'    # Directory name of the result

# Import the merged dataset
dataname = dirname_output + '/merged_data.xlsx'
with pd.ExcelFile(dataname) as data_merged:  
    df_ols = pd.read_excel(data_merged, credit_measure+'_合并数据')   # Read the first sheet

# Print the distribution of the credit score by year
print(df_ols.groupby('year')['社会信用环境指数'].describe())

# Log transform the regional GDP data
df_ols['log_地区生产总值元'] = np.log(df_ols['地区生产总值万元'])

# Print the distribution of the regional GDP per capita by year
print(df_ols.groupby('year')['人均地区生产总值元'].describe())
print(df_ols.loc[df_ols.groupby('year')['人均地区生产总值元'].idxmax(),['year','city','人均地区生产总值元']])  # The city with the highest GDP per capita in each year

# Create a scatter plot of regional GDP and regional credit environment
plt.rcParams['font.sans-serif'] = ['SimSun']    # Set the font style
plt.rcParams['axes.unicode_minus'] = False       # Display the negative sign
plt.rcParams['figure.figsize'] = 12,8    # Set the figure size
sns_plot = sns.relplot(y='人均地区生产总值元', x='社会信用环境指数', data=df_ols, 
            hue='year', palette='viridis', alpha=0.5,
            height=8, aspect=1.5) 
plt.show()
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)
sns_plot.savefig(os.path.join(result_dir, 'regional_GDP_credit_score.png'),dpi=300)

# Plot the relationship between regional GDP and regional credit environment by year
os.makedirs(result_dir + '/scatter', exist_ok=True)
yearlist = df_ols['year'].unique()    # Unique year values
for year in yearlist:
    # Create a scatter plot of regional GDP and regional credit environment
    sns_plot = sns.relplot(y='人均地区生产总值元', x='社会信用环境指数', data=df_ols[df_ols['year']==year], 
                alpha=0.5, height=8, aspect=1.5)
    plt.title(f"Year {year} scatter plot")
    sns_plot.savefig(os.path.join(result_dir + '/scatter', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.close()  # Close the figure to free memory

# Draw regplots by year
os.makedirs(result_dir + '/lmfit', exist_ok=True)
for year in yearlist:
    # Create a scatter plot of regional GDP and regional credit environment
    sns_plot = sns.lmplot(y='人均地区生产总值元', x='社会信用环境指数', data=df_ols[df_ols['year']==year], 
                order=1, ci=95, height=8, aspect=1.5, scatter_kws={'alpha':0.5})
    plt.title(f"Year {year} regplot")
    # sns_plot.savefig(os.path.join(result_dir + '/lmfit', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.savefig(os.path.join(result_dir + '/lmfit', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.close()  # Close the figure to free memory

# Plot residuals against fitted values
os.makedirs(result_dir + '/residuals', exist_ok=True)
for year in yearlist:
    # Create a scatter plot of regional GDP and regional credit environment
    sns_plot = sns.residplot(x='人均地区生产总值元', y='社会信用环境指数', data=df_ols[df_ols['year']==year], 
                order=1, scatter_kws={'alpha':0.5})
    plt.title(f"Year {year}")
    # sns_plot.savefig(os.path.join(result_dir + '/residuals', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.savefig(os.path.join(result_dir + '/residuals', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.close()  # Close the figure to free memory




############################################
### Analyze different credit dimensions  ###
############################################

# Set directories
dirname_output = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Data/Output'    # Directory name of the output data
dirname_result = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Results'    # Directory name of the result
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)

# Plot the evolution of aggregate credit ratings over time
plt.rcParams['font.sans-serif'] = ['SimSun']    # Set the font style
plt.rcParams['axes.unicode_minus'] = False      # Display the negative sign
plt.figure(figsize=(12,8))
g = sns.catplot(x='year', y='社会信用环境指数',data=df_ols,
                kind='box',height=10, aspect=0.8, 
                boxprops=dict(facecolor="none"))
plt.xticks(rotation=90, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('年份',size=30)
plt.ylabel('')
plt.title('社会信用环境指数',size=30)
plt.savefig(os.path.join(result_dir, '社会信用环境指数.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot the evolution of each credit dimension over time
print(df_ols.columns)
dimlist = ['信用制度指数','信用技术指数','信用应用指数','信用文化指数']
f, axes = plt.subplots(2,2,figsize=(25, 16))
axes = axes.flatten()  # Flatten the 2D axes array into 1D for easy indexing
for idx, dim in enumerate(dimlist):
    sns.boxplot(x='year', y=dim, data=df_ols, ax=axes[idx],
                boxprops=dict(facecolor="none"))
    axes[idx].set_xlabel('年份', fontsize=30)
    axes[idx].set_ylabel('')
    axes[idx].set_title(dim, fontsize=30)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, fontsize=20)
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=20)
plt.subplots_adjust(hspace=0.3, wspace=0.1)  # Adjust spacing between subplots
plt.savefig(os.path.join(result_dir, '社会信用维度指数.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot the evolution of each credit sub-dimension over time
dimlist = ['政府信用制度','市场信用制度','信息技术','传输技术','应用技术',
            '政务应用','商务应用','社会应用','司法应用','信用教育',
            '信用宣传','信用载体']

f, axes = plt.subplots(4,3,figsize=(25, 20))
axes = axes.flatten()  # Flatten the 2D axes array into 1D for easy indexing
for idx, dim in enumerate(dimlist):
    sns.boxplot(x='year', y=dim, data=df_ols, ax=axes[idx],
                boxprops=dict(facecolor="none"))
    axes[idx].set_xlabel('年份', fontsize=30)
    axes[idx].set_ylabel('')
    axes[idx].set_title(dim, fontsize=30)
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, fontsize=20)
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=20)
plt.subplots_adjust(hspace=0.7, wspace=0.1)  # Adjust spacing between subplots
plt.savefig(os.path.join(result_dir, '社会信用二级指标.png'), dpi=300, bbox_inches='tight')
plt.close()

# Report some summary statistics by year
print(df_ols.groupby('year')['社会信用环境指数'].agg([np.mean,np.median,np.std,np.min,np.max]))

# Report the summary statistics of each dimension by year
dimlist = ['信用制度指数','信用技术指数','信用应用指数','信用文化指数']
for dim in dimlist:
    print(dim)
    print(df_ols.groupby('year')[dim].agg([np.mean,np.median,np.std]))

# Report the summary statistics of each sub-dimension by year
dimlist = ['政府信用制度','市场信用制度','信息技术','传输技术','应用技术',
            '政务应用','商务应用','社会应用','司法应用','信用教育',
            '信用宣传','信用载体']
for dim in dimlist:
    print(dim)
    print(df_ols.groupby('year')[dim].agg([np.mean,np.median,np.std]))




###########################################################
### Analyze regional differences in credit environment  ###
###########################################################

dirname_output = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Data/Output'    # Directory name of the output data
dirname_result = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Results'    # Directory name of the result
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)

# Subset the credit score by region
print(df_ols.columns)
print(df_ols['province'].unique())

north_eastern = ['辽宁省','吉林省','黑龙江省']
eastern = ['北京市','天津市','河北省','上海市','江苏省','浙江省','福建省','山东省',
            '广东省','海南省']
middle = ['山西省','安徽省','江西省','河南省','湖北省','湖南省']
western = ['内蒙古自治区','广西壮族自治区','重庆市','四川省','贵州省','云南省','西藏自治区',
            '陕西省','甘肃省','青海省','宁夏回族自治区','新疆维吾尔自治区']

# Assign region for each province or city
region_dict = {}
for area in north_eastern:
    region_dict[area] = '东北地区'
for area in eastern:
    region_dict[area] = '东部地区'
for area in middle:
    region_dict[area] = '中部地区'
for area in western:
    region_dict[area] = '西部地区'
print(region_dict)

# Create a new column that specifies the region the city belongs to
df_ols['region'] = NaN
df_ols['region'] = df_ols['region'].fillna(df_ols['province'].map(region_dict))

# Produce a line plot that displays the evolution of credit scores across region
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='year', y='社会信用环境指数', data=df_ols,
            hue='region', style='region', markers=False, 
            linewidth=3, ci=None, palette=['#555555'])
ax.set_title('四大区域社会信用环境指数', fontsize=24)
ax.set_ylabel('')
ax.set_xlabel('年份', fontsize=16)
ax.set_xticks(np.linspace(df_ols['year'].min(),df_ols['year'].max(),
                          num=df_ols['year'].nunique()))
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=16)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
ax.legend(title='区域')
plt.savefig(os.path.join(result_dir, '四大区域社会信用环境指数.png'), dpi=300, bbox_inches='tight')

# Plot the evolution of each credit dimension over time
print(df_ols.columns)
dimlist = ['信用制度指数','信用技术指数','信用应用指数','信用文化指数']
f, axes = plt.subplots(2, 2, figsize=(25, 16))
axes = axes.flatten()  # Flatten the 2D axes array into 1D for easy indexing
gray_scale = ['#555555'] * 4
for idx, dim in enumerate(dimlist):
    sns.lineplot(x='year', y=dim, data=df_ols,  # Filter data for the current dimension
        hue='region', style='region', markers=False,
        linewidth=3, ci=None, palette=gray_scale,
        ax=axes[idx])  # Use the correct subplot
    legend = axes[idx].get_legend()
    if legend is not None:
        legend.set_title("区域")  # Change the legend title to "区域"
    # Set subplot labels and title
    axes[idx].set_xlabel('年份', fontsize=20)
    axes[idx].set_ylabel('')
    axes[idx].set_title(dim, fontsize=30)
    axes[idx].set_xticks(np.linspace(df_ols['year'].min(),df_ols['year'].max(),num=df_ols['year'].nunique()))
    axes[idx].tick_params(axis='x', rotation=90)  # Rotate x-axis labels
    axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=90, fontsize=20)
    axes[idx].set_yticklabels(axes[idx].get_yticklabels(), fontsize=20)
plt.subplots_adjust(hspace=0.3, wspace=0.1)
plt.savefig(os.path.join(result_dir, '四大区域社会信用维度指数.png'), dpi=300, bbox_inches='tight')
plt.close()

# Print the summary statistics of different regions
region_list = df_ols['region'].unique()
for area in region_list:
    print('\n\n' + area + '\n\n')
    print(df_ols[df_ols['region']==area].groupby('year')[['社会信用环境指数','信用制度指数','信用技术指数','信用应用指数','信用文化指数']].mean())




##########################################################
### Analyze each credit dimension's measuring variable ###
##########################################################

# Set directories
dirname_input = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Data/Input'    # Directory name of the input data
dirname_output = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Data/Output'    # Directory name of the output data
dirname_result = 'D:/国信研究院实习/社会信用对区域经济的影响/regional-credit-environment-economy/Results'    # Directory name of the result
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)

#### Font ####
plt.rcParams['font.sans-serif'] = ['SimSun']    # Set the font style
plt.rcParams['axes.unicode_minus'] = False      # Display the negative sign

# Import the measuring variable dataset
dataname = dirname_input + '/社会信用环境指数_三级指标0824.xlsx'
datafile = pd.ExcelFile(dataname)  
print("The sheets in the dataset: ", datafile.sheet_names)  # Print the sheet names
with pd.ExcelFile(dataname) as data_all:  
    df_all = pd.read_excel(data_all, sheet_name='Sheet1')   # Read the first sheet
    
df_all.rename({'省':'province','市':'city'}, axis=1, inplace=True)  # Rename the province and city variables

remove_list = ['id', 'year', 'province', 'city', 'region', '企业数量', '人口']
negative_list = ['政府债务率', '行政处罚数量', '失信被执行人企业数量', '被执行人数量', '失信人数量', '知识产权案件数']

# Pick only columns you want to standardize (exclude remove_list):
varlist = df_all.columns.to_list()
varlist = [col for col in varlist if col not in remove_list]

# Standardize the data
def standardize_data(df, varlist, negative_list):
    for col in varlist:
        # Skip if not numeric
        if df[col].dtype.kind not in 'bifc':
            continue
        
        col_min = df[col].min()
        col_max = df[col].max()
        
        # To avoid division by zero (in case col_min == col_max):
        if col_min == col_max:
            # e.g. set that column to a constant 0.5 (or something else) 
            # since there's no variation in data
            df[col] = 0.5
            continue
        
        if col not in negative_list:
            # “Higher is better” => scale 0.1 to 1.0
            df[col] = (df[col] - col_min) / (col_max - col_min) * 0.9 + 0.1
        else:
            # “Lower is better” => invert and scale 0.1 to 1.0
            df[col] = (col_max - df[col]) / (col_max - col_min) * 0.9 + 0.1
    return df

df_all = standardize_data(df_all, varlist, negative_list)

# Add the 'region' column
df_all['region'] = NaN
df_all['region'] = df_all['region'].fillna(df_all['province'].map(region_dict))  # Create a column called 'region'

fig_dir = os.path.join(result_dir, '三级指标')
os.makedirs(fig_dir, exist_ok=True)

# Plot the evolution of all measuring variables:
gray_scale = ['#555555'] * 4    # Create a gray palette
for var in varlist:
    f, ax = plt.subplots(1, 1, figsize=(20, 12))
    sns.lineplot(x='year', y=var, data=df_all,  # Filter data for the current dimension
        hue='region', style='region', markers=False,
        linewidth=3, ci=None, palette=gray_scale) 
    
    # Set x-tick
    xticks = np.linspace(df_all['year'].min(), df_all['year'].max(), num=df_all['year'].nunique(), dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, rotation=90, fontsize=26)

    # Set y-tick
    y_min, y_max = ax.get_ylim()
    num_yticks = 6 
    yticks = np.linspace(y_min, y_max, num=num_yticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.2f}" for tick in yticks], fontsize=26)
    ax.set_xlabel('年份', fontsize=32)
    ax.set_ylabel('', fontsize=30)
    ax.set_title(var, fontsize=40)
    
    # 1. Grab handles and labels from the current Axes
    handles, labels = ax.get_legend_handles_labels()

    # 2. Remove the old legend (if it exists)
    old_legend = ax.get_legend()
    if old_legend is not None:
        old_legend.remove()

    # 3. Create a new legend with a longer handlelength
    new_legend = ax.legend(
        handles, 
        labels,
        handlelength=1.5,      # Increase this value to lengthen dashed lines
        # loc='best',          # or "upper right", etc.
        title="区域",         # legend title
        title_fontsize=32,
        prop={'size': 32}    # text size for labels
        )
                
    filename = var + '.png'
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(os.path.join(fig_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()




#####################################################################
### Plot the credit environment ranking vs GDP per capita ranking ###
#####################################################################

# Import the merged dataset
dataname = dirname_output + '/merged_data.xlsx'
with pd.ExcelFile(dataname) as data_merged:  
    df_ols = pd.read_excel(data_merged, credit_measure+'_合并数据')   # Read the first sheet

# Obtain the rankings of regional credit ratings and GDP per capita
df_ols['credit_ranking'] = df_ols.groupby('year')['社会信用环境指数'].rank(ascending=True, method='max').astype(int)
df_ols['GDP_ranking'] = df_ols.groupby('year')['人均地区生产总值元'].rank(ascending=True, method='max').astype(int)

# Scatter plot
sns_plot = sns.relplot(y='credit_ranking', x='GDP_ranking', data=df_ols, 
            hue='year', palette='viridis', alpha=0.5,
            height=8, aspect=1.5) 
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)
sns_plot.savefig(os.path.join(result_dir, 'GDP_credit_ranking.png'),dpi=300)

# Scatter plot by year
os.makedirs(result_dir + '/scatter_ranking', exist_ok=True)
yearlist = df_ols['year'].unique()    # Unique year values
for year in yearlist:
    # Create a scatter plot of regional GDP and regional credit environment
    sns_plot = sns.relplot(x='GDP_ranking', y='credit_ranking', data=df_ols[df_ols['year']==year], 
                alpha=0.5, height=8, aspect=1.5)
    plt.title(f"Year {year} ranking scatter plot")
    sns_plot.savefig(os.path.join(result_dir + '/scatter_ranking', f'regional_GDP_credit_score_{year}.png'),dpi=300)
    plt.close()  # Close the figure to free memory

# Scatter plot of means
average_credit = df_ols.groupby('city')['社会信用环境指数'].mean()
average_GDP = df_ols.groupby('city')['人均地区生产总值元'].mean()
credit_ranking = average_credit.rank(ascending=True, method='max').astype(int)
GDP_ranking = average_GDP.rank(ascending=True, method='max').astype(int)
print(credit_ranking)
print(GDP_ranking)

plt.figure(figsize=(25,16))
plt.scatter(x=GDP_ranking, y=credit_ranking)
plt.xlabel('人均GDP排名', fontsize=20)
plt.ylabel('社会信用环境指数排名', fontsize=20)
plt.title('平均排名', fontsize=26)
result_dir = os.path.join(dirname_result, 'figures', credit_measure)
os.makedirs(result_dir, exist_ok=True)
plt.savefig(os.path.join(result_dir, 'average_GDP_credit_ranking.png'))
plt.close()




