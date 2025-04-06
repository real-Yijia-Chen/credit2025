Project Name: Measuring Credit Environment and Regional Economic Growth: An Empirical Analysis Based on Four Dimensions
Date: January 24, 2025

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

File Description

This folder contains two Python source files, each serving the following functions:
1. 'clean_data.py' – Reads data from an Excel file and fills in missing values.
2. 'exploratory_data_analysis.py' - Creates scatter plots and line charts showing the relationship between regional credit environment and regional per capita GDP.
3. 'regression_analysis.py' - Performs regression analysis and exports the results to an Excel file.
4. 'robustness_check.py' – Performs Friedman and Nemenyi tests on the ranking results of urban credit environments.

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Instructions for Running the Code

Before running the Python scripts mentioned above, in line 26 replace the placeholder "your_dirname" in the source files with the actual file path where the "credit2025" folder is located. Then, execute the corresponding script.

The credit environment measure can be set in line 31. Available options are:
1. 'expert' – Expert weighting method
2. 'principal_component' – Principal component analysis (PCA)
3. 'equal_weight' – Equal-weight method

The cleaned data is stored in: 'your_dirname/Data/Output/merged_data.xls'
Results of robustness check are stored as Excel files in: 'your_dirname/Results/regression_results'

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Explanation 

The Excel file '专家_regression_results.xls' contains four worksheets, each corresponding to one of the regression result tables in the paper:

1. Sheet1 corresponds to Table 2.
2. Sheet2 corresponds to Table 4.
3. Sheet3 corresponds to Table 5.
4. Sheet4 corresponds to Table 3.

The file 'expert_robustness_check.xls' contains two worksheets corresponding to the robustness check results in the paper:

1. Friedman test – Results from Table 6 (Friedman test)
2. Nemenyi test – Results from Table 7 (Nemenyi test)




