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




#########################################################################################
### Regression analysis: the economic impact of improving regional credit environment ###
#########################################################################################

# Set directories
dirname_output = dirname_main + '/Data/Output'    # Directory name of the output data
dirname_result = dirname_main + '/Results'    # Directory name of the result

# Import the merged dataset
dataname = dirname_output + '/merged_data.xlsx'
with pd.ExcelFile(dataname) as data_merged:  
    df_ols = pd.read_excel(data_merged, credit_measure+'_合并数据')   # Read the first sheet

# Generate dependent and control variables
df_ols['lnAGdp'] = np.log(df_ols['人均地区生产总值元'])     # Log GDP per capita
df_ols['Finadp'] = df_ols['地方财政一般预算内收入万元'] / df_ols['地方财政一般预算内支出万元']  # Ratio of fiscal revenue over fiscal spending
df_ols['Urban'] = np.log(df_ols['城区人口万人'] / df_ols['城区面积平方公里'])   # Log ratio of urban population over urban area
df_ols['Area'] = np.log(df_ols['行政区域土地面积平方公里'] / df_ols['户籍人口万人'])    # Log ratio of admin area over total population
df_ols['FA'] = (df_ols['固定资产投资亿元'] * (10**4)) / df_ols['地区生产总值万元']      # Ratio of fixed asset investment over GDP
df_ols['IM'] = df_ols['进口额_百万美元'] / df_ols['户籍人口万人 ']    # Import per capita
df_ols['EX'] = df_ols['出口额_百万美元'] / df_ols['户籍人口万人 ']    # Export per capita 
df_ols['MFDI'] = df_ols['FDI_百万美元'] / df_ols['户籍人口万人 ']    # Foreign direct investment per capita



# Winsorize the dataframe at 1% tails
def winsorize_dataframe(df, lower_quantile=0.01, upper_quantile=0.99):
    """
    Winsorize all numeric columns of a DataFrame by capping
    values below the specified lower_quantile and above the specified upper_quantile.
    """
    df_wins = df.copy()  # Don't mutate the original DataFrame
    numeric_cols = df_wins.select_dtypes(include=[np.number]).columns.tolist()
    
    # Drop 'year' and 'id' from the list of variables to be winsorized
    if 'year' in numeric_cols:
        numeric_cols.remove('year')
    if 'id' in numeric_cols:
        numeric_cols.remove('id')

    # Define a function to apply winsorization to each group
    def winsorize_group(group):
        for col in numeric_cols:
            lower_bound = group[col].quantile(lower_quantile)
            upper_bound = group[col].quantile(upper_quantile)
            group[col] = group[col].clip(lower=lower_bound, upper=upper_bound)
        return group

    # Apply the winsorization function to each year group
    df_wins = df_wins.groupby('year').apply(winsorize_group)

    # If groupby adds a hierarchical index, reset it to maintain the original structure
    if isinstance(df_wins.index, pd.MultiIndex):
        df_wins.reset_index(drop=True, inplace=True)

    return df_wins
df_ols = winsorize_dataframe(df_ols)
df_ols.sort_values(['id','year'],inplace=True)  # Sort the observations by 'id' and 'year'
df_ols.reset_index(drop=True, inplace=True)     # Reset index

#################### Explore the relationship between GDP per capita and regional credit environment ####################

# Define a function that runs and stores the results of four regression models
def run_models1(df_ols):
    # Create a dictionary to store results
    results = {}

    # Model 1: Simple univariate linear regression
    global model1
    model1 = ols('lnAGdp ~ 社会信用环境指数', data=df_ols).fit()
    results['Model 1'] = model1.summary()    # Coefficients and stats table

    # Model 2：Add time and province fixed effects but without control variables
    global model2
    model2 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 2'] = model2.summary

    # Model 3: Add control variables but without fixed effects
    global model3
    model3 = ols('lnAGdp ~ 社会信用环境指数 + Finadp + Urban + Area + FA', data=df_ols).fit()
    results['Model 3'] = model3.summary()

    # Model 4：Add time and province fixed effects and control variables
    global model4
    model4 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 4'] = model4.summary

    return results

# Run the models and store results
regression_results = run_models1(df_ols)

# Print the regression results
for key, value in regression_results.items():
    print(f"\n\n {key} \n\n {value} \n\n")
    print(f"Type of results: {type(value)}")

# Store the results using 'stargazer'
model_objects = run_models1(df_ols)
reg_output = Stargazer([model1, model2, model3, model4])

# Reorder covariates:
desired_order = ['社会信用环境指数','Finadp','Urban','Area','FA','Intercept']
reg_output.covariate_order(desired_order)

# Rename covariates:
reg_output.rename_covariates({'社会信用环境指数':'Credit'})

# Force stargazer to display p-value instead of standard error
for model_dict in reg_output.model_data:
    model_dict["cov_std_err"] = model_dict["p_values"]

# Convert the regression results to an HTML file
table_html = reg_output.render_html()

# Convert the HTML to a Pandas dataframe and save it in an excel file
df_reg1 = pd.read_html(table_html)[0]

# Save the regression results to an excel file
regression_dir = os.path.join(dirname_result, 'regression_results', credit_measure)
os.makedirs(regression_dir, exist_ok=True)
regression_file = credit_measure + '_regression_results.xlsx'

# Save the dataset
with pd.ExcelWriter(os.path.join(regression_dir, regression_file), engine='xlsxwriter') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_reg1.to_excel(writer, sheet_name='Sheet1', index=False)    # Save the dataset




#########################################################################
### Regression analysis: the economic impact of each credit dimension ###
#########################################################################

# Define a function that runs and stores the results of four regression models
def run_models2(df_ols):
    # Create a dictionary to store results
    results = {}

    # Model 5：Estimate the economic impact of credit institutions
    global model5
    model5 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用制度指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 5'] = model5.summary

    # Model 6：Estimate the economic impact of credit technology
    global model6
    model6 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用技术指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 6'] = model6.summary

    # Model 7：Estimate the economic impact of credit application
    global model7
    model7 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用应用指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 7'] = model7.summary

    # Model 8：Estimate the economic impact of credit culture
    global model8
    model8 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用文化指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 8'] = model8.summary

    return results

# Run the models and store results
regression_results = run_models2(df_ols)

# Print the regression results
for key, value in regression_results.items():
    print(f"\n\n {key} \n\n {value} \n\n")
    print(f"Type of results: {type(value)}")

# Store the results using 'stargazer'
model_objects = run_models2(df_ols)
reg_output = Stargazer([model5, model6, model7, model8])

# Reorder covariates:
desired_order = ['信用制度指数','信用技术指数','信用应用指数','信用文化指数','Finadp','Urban','Area','FA','Intercept']
reg_output.covariate_order(desired_order)

# Rename covariates:
reg_output.rename_covariates({'信用制度指数':'Inst','信用技术指数':'Tech',
                                '信用应用指数':'App','信用文化指数':'Culture'})

# Force stargazer to display p-value instead of standard error
for model_dict in reg_output.model_data:
    model_dict["cov_std_err"] = model_dict["p_values"]

# Convert the regression results to an HTML file
table_html = reg_output.render_html()

# Convert the HTML to a Pandas dataframe and save it in an excel file
df_reg2 = pd.read_html(table_html)[0]

# Save the regression results to an excel file
regression_dir = os.path.join(dirname_result, 'regression_results', credit_measure)
os.makedirs(regression_dir, exist_ok=True)
regression_file = credit_measure + '_regression_results.xlsx'

# Save the dataset
with pd.ExcelWriter(os.path.join(regression_dir, regression_file), engine='openpyxl', mode='a') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_reg2.to_excel(writer, sheet_name='Sheet2', index=False)    # Save the dataset




#############################################
### Regression analysis: lagged variables ###
#############################################

# Generate lagged variables
varlist = ['社会信用环境指数','信用制度指数','信用技术指数','信用应用指数','信用文化指数']  # Generate a variable list
for var in varlist:
    lag1_name = var + '_lag1'
    lag2_name = var + '_lag2'
    df_ols[lag1_name] = df_ols.groupby('city')[var].shift(1)    # 1-period lag
    df_ols[lag2_name] = df_ols.groupby('city')[var].shift(2)    # 2-period lag

print(df_ols[['city','year','社会信用环境指数','社会信用环境指数_lag1','社会信用环境指数_lag2']].head(40))

# Run regressions for each variable and its laggs in the variable list
def run_models3(df_ols):
    # Create a dictionary to store results
    results = {}

    # Model 9：Estimate the economic impact of credit environment and its laggs
    global model9
    model9 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + 社会信用环境指数_lag1 + 社会信用环境指数_lag2 + \
                                    Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 9'] = model9.summary

    # Model 10：Estimate the economic impact of credit institutions and its laggs
    global model10
    model10 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用制度指数 + 信用制度指数_lag1 + 信用制度指数_lag2 + \
                                    Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 10'] = model10.summary

    # Model 11：Estimate the economic impact of credit technology and its laggs
    global model11
    model11 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用技术指数 + 信用技术指数_lag1 + 信用技术指数_lag2 + \
                                    Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 11'] = model11.summary

    # Model 12：Estimate the economic impact of credit technology and its laggs
    global model12
    model12 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用应用指数 + 信用应用指数_lag1 + 信用应用指数_lag2 + \
                                    Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 12'] = model12.summary

    # Model 13：Estimate the economic impact of credit technology and its laggs
    global model13
    model13 = PanelOLS.from_formula('lnAGdp ~ 1 + 信用文化指数 + 信用文化指数_lag1 + 信用文化指数_lag2 + \
                                    Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.set_index(['province','year'])).fit(cov_type='clustered', cluster_entity=True, cluster_time=True)
    results['Model 13'] = model13.summary

    return results

# Run the models and store results
regression_results = run_models3(df_ols)

# Print the regression results
for key, value in regression_results.items():
    print(f"\n\n {key} \n\n {value} \n\n")
    print(f"Type of results: {type(value)}")

# Store the results using 'stargazer'
model_objects = run_models3(df_ols)
reg_output = Stargazer([model9, model10, model11, model12, model13])

# Reorder covariates:
desired_order = ['社会信用环境指数','社会信用环境指数_lag1','社会信用环境指数_lag2','信用制度指数','信用制度指数_lag1','信用制度指数_lag2',\
                '信用技术指数','信用技术指数_lag1','信用技术指数_lag2','信用应用指数','信用应用指数_lag1','信用应用指数_lag2',\
                '信用文化指数','信用文化指数_lag1','信用文化指数_lag2','Finadp','Urban','Area','FA','Intercept']
reg_output.covariate_order(desired_order)

# Rename covariates:
reg_output.rename_covariates({'社会信用环境指数':'Credit','社会信用环境指数_lag1':'Credit_lag1','社会信用环境指数_lag2':'Credit_lag2',\
                            '信用制度指数':'Inst','信用制度指数_lag1':'Inst_lag1','信用制度指数_lag2':'Inst_lag2',\
                            '信用技术指数':'Tech','信用技术指数_lag1':'Tech_lag1','信用技术指数_lag2':'Tech_lag2',\
                            '信用应用指数':'App','信用应用指数_lag1':'App_lag1','信用应用指数_lag2':'App_lag2',\
                            '信用文化指数':'Culture','信用文化指数_lag1':'Culture_lag1','信用文化指数_lag2':'Culture_lag2'})

# Force stargazer to display p-value instead of standard error
for model_dict in reg_output.model_data:
    model_dict["cov_std_err"] = model_dict["p_values"]

# Convert the regression results to an HTML file
table_html = reg_output.render_html()

# Convert the HTML to a Pandas dataframe and save it in an excel file
df_reg3 = pd.read_html(table_html)[0]

# Save the regression results to an excel file
regression_dir = os.path.join(dirname_result, 'regression_results', credit_measure)
os.makedirs(regression_dir, exist_ok=True)
regression_file = credit_measure + '_regression_results.xlsx'

# Save the dataset
with pd.ExcelWriter(os.path.join(regression_dir, regression_file), engine='openpyxl', mode='a') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_reg3.to_excel(writer, sheet_name='Sheet3', index=False)    # Save the dataset




###################################################
### Regression analysis: regional heterogeneity ###
###################################################

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

# Create a new column that specifies the region that the city belongs to
df_ols['region'] = NaN
df_ols['region'] = df_ols['region'].fillna(df_ols['province'].map(region_dict))

# Regress GDP per capita on credit environment, using subsets of the data set (subset by region)

# Define a function that runs and stores the results of four regression models
def run_models4(df_ols):
    # Create a dictionary to store results
    results = {}

    # Model 14：Add time and province fixed effects and control variables
    global model14
    model14 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.loc[df_ols['region']=='东北地区'].set_index(['province','year'])).fit(\
                                        cov_type='clustered', cluster_entity=True, cluster_time=True) # Use observations belonging to northeastern China
    results['Model 14'] = model14.summary

    # Model 15：Add time and province fixed effects and control variables
    global model15
    model15 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.loc[df_ols['region']=='东部地区'].set_index(['province','year'])).fit(\
                                        cov_type='clustered', cluster_entity=True, cluster_time=True) # Use observations belonging to eastern China
    results['Model 15'] = model15.summary

    # Model 16：Add time and province fixed effects and control variables
    global model16
    model16 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.loc[df_ols['region']=='中部地区'].set_index(['province','year'])).fit(\
                                        cov_type='clustered', cluster_entity=True, cluster_time=True) # Use observations belonging to middle China
    results['Model 16'] = model16.summary

    # Model 17：Add time and province fixed effects and control variables
    global model17
    model17 = PanelOLS.from_formula('lnAGdp ~ 1 + 社会信用环境指数 + Finadp + Urban + Area + FA + EntityEffects + TimeEffects',
                                    data=df_ols.loc[df_ols['region']=='西部地区'].set_index(['province','year'])).fit(\
                                        cov_type='clustered', cluster_entity=True, cluster_time=True) # Use observations belonging to western China
    results['Model 17'] = model17.summary

    return results

# Run the models and store results
regression_results = run_models4(df_ols)

# Print the regression results
for key, value in regression_results.items():
    print(f"\n\n {key} \n\n {value} \n\n")
    print(f"Type of results: {type(value)}")

# Store the results using 'stargazer'
model_objects = run_models4(df_ols)
reg_output = Stargazer([model14, model15, model16, model17])

# Reorder covariates:
desired_order = ['社会信用环境指数','Finadp','Urban','Area','FA','Intercept']
reg_output.covariate_order(desired_order)

# Rename covariates:
reg_output.rename_covariates({'社会信用环境指数':'Credit'})

# Force stargazer to display p-value instead of standard error
for model_dict in reg_output.model_data:
    model_dict["cov_std_err"] = model_dict["p_values"]

# Convert the regression results to an HTML file
table_html = reg_output.render_html()

# Convert the HTML to a Pandas dataframe and save it in an excel file
df_reg4 = pd.read_html(table_html)[0]

# Save the regression results to an excel file
regression_dir = os.path.join(dirname_result, 'regression_results', credit_measure)
os.makedirs(regression_dir, exist_ok=True)
regression_file = credit_measure + '_regression_results.xlsx'

# Save the dataset
with pd.ExcelWriter(os.path.join(regression_dir, regression_file), engine='openpyxl', mode='a') as writer:  # Create an ExcelWriter object
    # Save the dataset
    df_reg4.to_excel(writer, sheet_name='Sheet4', index=False)    # Save the dataset

