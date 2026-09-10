import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
df= pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/CarPrice.csv")

print(df.head())
x = df[['enginesize']]
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Training data size:", len(x_train))
print("Testing data size:", len(x_test))
print("X shape:", x.shape)
print("Y shape:", y.shape)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

plt.scatter(x_test, y_test, color='blue', label='Actual Price')
plt.plot(x_test, y_pred, color='red', linewidth=2, label='Predicted Price')
plt.title('car price prediction: Engine Size vs Price')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()