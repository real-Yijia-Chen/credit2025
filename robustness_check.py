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




##########################################################################
### Test the consistency between different credit environment measures ###
##########################################################################

# Set directories
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
dirname_result = dirname_main + '/Results'    # Directory name of the result

# Import the datasets
dataname = dirname_output + '/credit_score.xlsx'
with pd.ExcelFile(dataname) as data_credit:  
    df_expert = pd.read_excel(data_credit, sheet_name='专家', usecols=['id','year','city','社会信用环境指数_专家'])   # Read the sheet corresponding to expert grading
    df_weights = pd.read_excel(data_credit, sheet_name='等权', usecols=['id','year','city','社会信用环境指数_等权'])   # Read the sheet corresponding to equal weights
    df_PCA = pd.read_excel(data_credit, sheet_name='主成分', usecols=['id','year','city','社会信用环境指数_主成分'])   # Read the sheet corresponding to principal component analysis

# Sort each dataset by the credit score and create a ranking column
df_expert.rename(columns={'社会信用环境指数_专家':'credit_expert'},inplace=True)
df_expert['ranking_expert'] = df_expert.groupby('year')['credit_expert'].rank(ascending=False, method='max').astype(int)
df_expert.sort_values(by=['year','id'], ascending=[True,True],inplace=True)

df_weights.rename(columns={'社会信用环境指数_等权':'credit_weights'},inplace=True)
df_weights['ranking_weights'] = df_weights.groupby('year')['credit_weights'].rank(ascending=False, method='max').astype(int)
df_weights.sort_values(by=['year','id'], ascending=[True,True],inplace=True)

df_PCA.rename(columns={'社会信用环境指数_主成分':'credit_PCA'},inplace=True)
df_PCA['ranking_PCA'] = df_PCA.groupby('year')['credit_PCA'].rank(ascending=False, method='max').astype(int)
df_PCA.sort_values(by=['year','id'], ascending=[True,True],inplace=True)

# Merge the three datasets
df_credit = df_expert.merge(df_weights, on=['year','id','city'], how='left', validate='one_to_one').merge(df_PCA, 
                            on=['year','id','city'], how='left', validate='one_to_one')
print(df_credit.head(10))

# Initialize the result dictionary with empty lists
Friedman_dict = {
    'year': [],
    'test_statistics': [],
    'p_value': [],
    'p_value_corrected': []
}

# Conduct Friedman test by year and adjust the p-values using Bonferroni correction
num_years = df_credit['year'].nunique()
for tt in df_credit['year'].unique():
    # Conduct Friedman test by year
    Friedman_sample = df_credit.loc[df_credit['year']==tt, ['ranking_expert','ranking_weights','ranking_PCA']]
    res = friedmanchisquare(Friedman_sample.iloc[:,0], Friedman_sample.iloc[:,1], Friedman_sample.iloc[:,2])
    Friedman_result = [res.statistic, res.pvalue, min(res.pvalue * num_years, 1)]   # In the last entry, apply Bonferroni correction mannually

    # Store the result in a dictionary
    Friedman_dict['year'].append(tt)
    Friedman_dict['test_statistics'].append(Friedman_result[0])
    Friedman_dict['p_value'].append(Friedman_result[1])
    Friedman_dict['p_value_corrected'].append(Friedman_result[2]) 

# Create a dataframe using the result dictionary
df_Friedman = pd.DataFrame.from_dict(Friedman_dict)
print(df_Friedman)  # The rankings of regional credit environment have no significant differences except for year 2021

# Conduct a post hoc Nemenyi test for year 2021
posthoc_sample = df_credit.loc[df_credit['year']==2021, ['ranking_expert','ranking_weights','ranking_PCA']]
df_nemenyi = sp.posthoc_nemenyi_friedman(posthoc_sample)

# Save the results to an excel file
regression_dir = os.path.join(dirname_result, 'regression_results', credit_measure)
os.makedirs(regression_dir, exist_ok=True)
regression_file = credit_measure + '_robustness_check.xlsx'

with pd.ExcelWriter(os.path.join(regression_dir, regression_file)) as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_Friedman.to_excel(writer, sheet_name='Friedman test', index=False)    # Save the dataset
    df_nemenyi.to_excel(writer, sheet_name='Nemenyi test', index=False)    # Save the dataset

