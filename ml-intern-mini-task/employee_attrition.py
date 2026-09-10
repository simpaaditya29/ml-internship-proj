import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

url = "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/emp_attrition.csv"
df = pd.read_csv(url)

print(df.head())
le = LabelEncoder()
df['Attrition'] = le.fit_transform(df['Attrition'])
x = df[['Age', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
y = df['Attrition']

print("\nAFTER ENCODING:")
print(df[['Age', 'Attrition']].head())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("model trained and predictions made")

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")

print("\n---COnfusion Matrix---")
print(confusion)