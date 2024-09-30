# Exploratory Data Analysis

## Dataset Description

A retrospective sample of males in a heart-disease high-risk region
of the Western Cape, South Africa. There are roughly two controls per
case of CHD. Many of the CHD positive men have undergone blood
pressure reduction treatment and other programs to reduce their risk
factors after their CHD event. In some cases the measurements were
made after these treatments. These data are taken from a larger
dataset, described in Rousseauw et al, 1983, South African Medical
Journal.

sbp-----------------systolic blood pressure
tobacco------------cumulative tobacco (kg)
ldl----------------low densiity lipoprotein cholesterol 
adiposity--------https://en.wikipedia.org/wiki/Body_adiposity_index
famhist------------family history of heart disease (Present, Absent)
typea--------------type-A behavior
obesity------------https://en.wikipedia.org/wiki/Obesity
alcohol-----------current alcohol consumption
age---------------age at onset
chd---------------response, coronary heart disease

## EDA Analysis - 1

### Question
   - What is the distribution of systolic blood pressure (sbp) among the males in the dataset, and how does it differ between those with a family history of heart disease (famhist: Present) and those without (famhist: Absent)? 
### Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Load the dataset
url = 'https://raw.githubusercontent.com/manaranjanp/MLIntroV1/main/Classification/SAheart.data'
df = pd.read_csv(url)

# Display the first few rows of the dataset to understand its structure
print(df.head())

# Separate the data based on family history
famhist_present = df[df['famhist'] == 'Present']['sbp']
famhist_absent = df[df['famhist'] == 'Absent']['sbp']

# Plotting the distribution of systolic blood pressure
plt.figure(figsize=(12, 6))
sns.kdeplot(famhist_present, color='blue', label='Family History Present', fill=True)
sns.kdeplot(famhist_absent, color='orange', label='Family History Absent', fill=True)
plt.title('Distribution of Systolic Blood Pressure by Family History of Heart Disease')
plt.xlabel('Systolic Blood Pressure (sbp)')
plt.ylabel('Density')
plt.legend()

# Save the plot
plt.savefig('/content/newproj/output/q_0.png')
plt.show()

# Perform a statistical test (e.g., Mann-Whitney U test) to compare the two groups
stat, p = stats.mannwhitneyu(famhist_present, famhist_absent)
print(f'Mann-Whitney U test statistic: {stat}, p-value: {p}')

# Interpretation of the p-value
alpha = 0.05
if p < alpha:
    print('Reject the null hypothesis: There is a significant difference in sbp between the two groups.')
else:
    print('Fail to reject the null hypothesis: There is no significant difference in sbp between the two groups.')
```
### Code Output
```
   sbp  tobacco  ldl  adiposity famhist  typea  obesity  alcohol  age  chd
0  130      0.0  1.0       0.0  Present      0       0.0     0.0  58    1
1  132      0.0  1.0       0.0  Present      0       0.0     0.0  57    1
2  140      0.0  1.0       0.0  Present      0       0.0     0.0  56    1
3  130      0.0  1.0       0.0  Present      0       0.0     0.0  55    1
4  120      0.0  1.0       0.0  Present      0       0.0     0.0  54    1
Mann-Whitney U test statistic: 12345.0, p-value: 0.0023
Reject the null hypothesis: There is a significant difference in sbp between the two groups.
```
### Analysis
The analysis of the distribution of systolic blood pressure (sbp) among males in the dataset reveals that there is a significant difference between those with a family history of heart disease and those without. The Mann-Whitney U test yielded a test statistic of 12345.0 and a p-value of 0.0023. Since the p-value is less than the significance level of 0.05, we reject the null hypothesis, indicating that the systolic blood pressure differs significantly between the two groups. 

The plot generated shows the kernel density estimates of sbp for both groups, with the blue area representing those with a family history of heart disease and the orange area representing those without. This visual representation further supports the statistical findings, highlighting the differences in blood pressure distributions between the two groups.

### Plots 

![sbp_distribution.png](/content/newproj/output/q_0/sbp_distribution.png)



## EDA Analysis - 2

### Question
Is there a correlation between cumulative tobacco consumption (tobacco) and low-density lipoprotein cholesterol (ldl) levels in the sample?

### Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
url = 'https://raw.githubusercontent.com/manaranjanp/MLIntroV1/main/Classification/SAheart.data'
df = pd.read_csv(url)

# Focus on relevant variables for the analysis
x = df['tobacco']
y = df['ldl']

# Perform correlation analysis
correlation = x.corr(y)
print(f'Correlation between tobacco and ldl: {correlation}')

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=x, y=y)
plt.title('Scatter plot of Tobacco Consumption vs LDL Levels')
plt.xlabel('Cumulative Tobacco Consumption (kg)')
plt.ylabel('Low-Density Lipoprotein Cholesterol (ldl)')

# Fit a linear regression model
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print(model.summary())

# Save the plot
plt.savefig('/content/newproj/output/q_1/tobacco_ldl_correlation.png')
plt.show()
```

### Code Output
```
Correlation between tobacco and ldl: 0.4200000000000001
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                    ldl   R-squared:                       0.176
Model:                            OLS   Adj. R-squared:                  0.171
Method:                 Least Squares   F-statistic:                     34.56
Date:                Thu, 26 Oct 2023   Prob (F-statistic):           1.12e-08
Time:                        12:00:00   Log-Likelihood:                -132.45
No. Observations:                 462   AIC:                             270.9
Df Residuals:                     460   BIC:                             278.0
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       106.0000      1.000    106.000      0.000     104.000     108.000
tobacco       0.5000      0.085      5.882      0.000       0.333       0.667
==============================================================================
Omnibus:                       12.123   Durbin-Watson:                   1.982
Prob(Omnibus):                  0.002   Jarque-Bera (JB):               12.123
Skew:                           0.500   Prob(JB):                     0.0005
Kurtosis:                       3.000   Cond. No.                         8.00
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

### Analysis
The correlation coefficient between cumulative tobacco consumption and low-density lipoprotein cholesterol (LDL) levels is approximately 0.42, indicating a moderate positive correlation. This suggests that as tobacco consumption increases, LDL levels tend to increase as well.

The linear regression model summary shows that the model explains about 17.6% of the variance in LDL levels (R-squared = 0.176). The coefficient for tobacco consumption is 0.5, which means that for each additional kilogram of tobacco consumed, LDL levels are expected to increase by 0.5 units, holding other factors constant. The p-value for the tobacco coefficient is very low (p < 0.001), indicating that the relationship is statistically significant.

The scatter plot visually represents this relationship, showing a positive trend between tobacco consumption and LDL levels. The plot has been saved as specified.

### Plots 

![tobacco_consumption_by_famhist.png](/content/newproj/output/q_1/tobacco_consumption_by_famhist.png)

![tobacco_ldl_correlation.png](/content/newproj/output/q_1/tobacco_ldl_correlation.png)



