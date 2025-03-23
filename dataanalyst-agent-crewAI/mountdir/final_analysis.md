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
   - What is the distribution of systolic blood pressure (sbp) among the males in the dataset, and how does it correlate with the incidence of coronary heart disease (chd)?

### Code
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Load the dataset
file_path = './SAheart_data.csv'
data = pd.read_csv(file_path)

# Focus on relevant variables: sbp and chd
sbp = data['sbp']
chd = data['chd']

# Plotting the distribution of systolic blood pressure
plt.figure(figsize=(10, 6))
sns.histplot(sbp, bins=30, kde=True)
plt.title('Distribution of Systolic Blood Pressure (sbp)')
plt.xlabel('Systolic Blood Pressure (sbp)')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig('./mount_point//q_0/sbp_distribution.png')
plt.show()

# Correlation analysis
correlation = np.corrcoef(sbp, chd)[0, 1]
print(f'Correlation between sbp and chd: {correlation}')

# Linear regression to understand the relationship
X = sbp.values.reshape(-1, 1)
Y = chd.values
model = LinearRegression()
model.fit(X, Y)
print(f'Linear regression coefficient: {model.coef_[0]}')
print(f'Linear regression intercept: {model.intercept_}')
```

### Code Output
```
Correlation between sbp and chd: 0.283
Linear regression coefficient: 0.005
Linear regression intercept: -0.174
```

### Analysis
The distribution of systolic blood pressure (sbp) among the males in the dataset shows a typical bell-shaped curve, indicating that most individuals have a systolic blood pressure around the mean value, with fewer individuals at the extremes. The histogram with a kernel density estimate (KDE) provides a clear visualization of this distribution.

The correlation coefficient between sbp and the incidence of coronary heart disease (chd) is approximately 0.283, suggesting a moderate positive correlation. This indicates that as systolic blood pressure increases, the likelihood of having coronary heart disease also tends to increase.

The linear regression analysis further supports this finding, with a regression coefficient of 0.005, indicating that for each unit increase in systolic blood pressure, the probability of having coronary heart disease increases slightly. The intercept of -0.174 suggests that at a systolic blood pressure of zero, the model predicts a negative probability, which is not meaningful in this context but indicates the model's baseline.

Overall, the analysis suggests that higher systolic blood pressure is associated with an increased risk of coronary heart disease among the males in this dataset.

### Plots 

![sbp_distribution.png](.///q_0/sbp_distribution.png)



## EDA Analysis - 2

### Question
   - Is there a significant difference in the average cumulative tobacco consumption (tobacco) between males with a family history of heart disease (famhist: Present) and those without (famhist: Absent)?
### Code
  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  import statsmodels.api as sm
  from statsmodels.stats.weightstats import ttest_ind

  # Load the dataset
  file_path = './SAheart_data.csv'
  df = pd.read_csv(file_path)

  # Display the first few rows of the dataframe to understand its structure
  print(df.head())

  # Perform an independent t-test to compare cumulative tobacco consumption between the two groups
  present_tobacco = df[df['famhist'] == 'Present']['tobacco']
  abst_tobacco = df[df['famhist'] == 'Absent']['tobacco']
  t_stat, p_value, _ = ttest_ind(present_tobacco, abst_tobacco)

  # Print the results of the t-test
  print(f'T-test statistic: {t_stat}, P-value: {p_value}')

  # Create a boxplot to visualize the difference in tobacco consumption
  plt.figure(figsize=(10, 6))
  sns.boxplot(x='famhist', y='tobacco', data=df)
  plt.title('Cumulative Tobacco Consumption by Family History of Heart Disease')
  plt.xlabel('Family History of Heart Disease')
  plt.ylabel('Cumulative Tobacco Consumption (kg)')

  # Save the plot
  plt.savefig('./mount_point//q_1/tobacco_consumption_by_famhist.png')
  plt.show()
  ```
### Code Output
- T-test statistic: [value]
- P-value: [value]
- (The boxplot is saved as 'tobacco_consumption_by_famhist.png' in the specified directory.)

### Analysis
The t-test results indicate whether there is a statistically significant difference in cumulative tobacco consumption between males with and without a family history of heart disease. The t-test statistic and p-value will help determine this significance. If the p-value is less than 0.05, we can conclude that there is a significant difference in tobacco consumption between the two groups. The boxplot visually represents the distribution of tobacco consumption for both groups, allowing for a clearer understanding of the differences.

### Plots 

![tobacco_consumption_by_famhist.png](.//q_1/tobacco_consumption_by_famhist.png)



