import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the data
df = sns.load_dataset('titanic')

# 2. Drop the mostly empty 'deck' column
df = df.drop(columns=['deck'])

# 3. Fill the missing ages with the median age
df['age'] = df['age'].fillna(df['age'].median())

# 4. Print the final count of missing values
print(df.isnull().sum())

sns.countplot(data=df, x= 'survived')

plt.show()