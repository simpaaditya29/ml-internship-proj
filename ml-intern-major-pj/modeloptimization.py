import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.tree import DecisionTreeClassifier

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

x = df.drop('Outcome', axis=1)
y = df['Outcome']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
}

print("training and tuning random forest")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid.fit(x_train, y_train)

print("\n--- random forest optimization---")
print("best settings:", rf_grid.best_params_)
print("best cross-validation accuracy: {:.2f}%".format(rf_grid.best_score_ * 100))

dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
    }
print("\ntraining and tuning decision tree")
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid.fit(x_train, y_train)

print("\n--- decision tree optimization---")
print("best settings:", dt_grid.best_params_)
print("best cross-validation accuracy: {:.2f}%".format(dt_grid.best_score_ * 100))

rf_test_acc = accuracy_score(y_test, rf_grid.predict(x_test))
dt_test_acc = accuracy_score(y_test, dt_grid.predict(x_test))

print("\n--- test accuracy ---")
print("random forest test accuracy: {:.2f}%".format(rf_test_acc * 100))
print("decision tree test accuracy: {:.2f}%".format(dt_test_acc * 100))