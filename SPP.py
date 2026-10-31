import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #importing the dataset
df = pd.read_csv('DATA.csv')
#printing the first 5 rows of the dataset
print(df.head())
#data ni vigat
df.info()
print(df.describe()) 
print("data types:\n", df.dtypes)


#unti = 1 & 2 copy
print("\nQuantitative Data: int64 and float64")
quantitative = df.select_dtypes(include=['int64', 'float64']) 
print("Quantitative Data:", quantitative)
print("Quantitative column labels:\n", quantitative.columns)

print("\nQualitative Data: object and category")
qualitative = df.select_dtypes(include=['object', 'category']) 
print("Qualitative Data:", qualitative)
print("Qualitative column labels:\n", qualitative.columns) 

qty_df = df.select_dtypes(include=['int64', 'float64']) #Quantitative dataframe 
print("\nSkewness:\n", qty_df.skew()) #skewness value 
print("\nKurtosis:\n", qty_df.kurt()) #kurtosis vlaue 




#chart for quantitative data
#histogram for quantitative data
plt.hist(df["Sales"], bins=10)
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")
plt.show()

#boxplot for quantitative data
sns.boxplot(x=df["Profit"])
plt.title("Profit Box Plot")
plt.show()

#pie chart for qualitative data
df["Product Name"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Product Name Distribution")
plt.ylabel("")
plt.show()

#bar chart for qualitative data
df["Area"].value_counts().plot(kind="bar")
plt.xlabel("Area")
plt.ylabel("Count")
plt.title("Area-wise Count")
plt.show()

#correlation matrix for quantitative data
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

#skewness and kurtosis for quantitative data
plt.figure()
pro=df.Sales
sns.histplot(pro, kde=True)
plt.title('Sales Distribution')
plt.xlabel('Sales')
plt.ylabel('Profit')
plt.show()

# Linear Regression (Unit 2)
from sklearn.linear_model import LinearRegression
df_clean = df.dropna(subset=['Sales', 'Profit'])

X = df_clean[['Sales']]
y = df_clean['Profit']

model = LinearRegression()
model.fit(X, y)
m = model.coef_[0]
b = model.intercept_
r2 = model.score(X, y)

regression_results = f"Equation: Profit = {m:.2f} * Sales + {b:.2f}\nR² value: {r2:.4f}"

# Save Regression Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.title("Simple Linear Regression: Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.legend()
plt.savefig('regression_chart.png')
plt.show()



#report generation using sweetviz
import sweetviz as sv

report = sv.analyze(df)
report.show_html("Sweetviz_Report.html")

import dtale

d = dtale.show(df)


# from AutoViz_Class import AutoViz_Class
# AV = AutoViz_Class()

# AV.AutoViz(
#     "Stor_Profit_Pridition.csv",
#     depVar="Result"
# )



# Dependent Variable
y = df["Profit"]

# Independent Variables
X = df[["Sales"]]

print("Dependent Variable:")
print(y)

print("\nIndependent Variables:")
print(X)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting Data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)




from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create Linear Regression Model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("LINEAR REGRESSION MODEL")

print("Model Training Completed Successfully!")


y_pred = model.predict(X_test)

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)



print("\nCoefficient:", model.coef_[0])
print("Intercept:", model.intercept_)




mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MODEL EVALUATION")

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R² Score:", r2)



print("\nRegression Equation:")

print(
    f"Profit = {model.coef_[0]:.2f} * Sales + "
    f"{model.intercept_:.2f}"
)



plt.figure(figsize=(8, 5))

plt.scatter(
    X_test,
    y_test,
    label="Actual Values"
)

plt.plot(
    X_test,
    y_pred,
    label="Predicted Values"
)

plt.xlabel("Sales")
plt.ylabel("Profit")

plt.title("Linear Regression - Sales vs Profit")

plt.legend()

plt.show()


df_clean = df.dropna(subset=["Sales", "Profit"])

X = df_clean[["Sales"]]      # Independent Variable
y = df_clean["Profit"]       # Dependent Variable


# TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# BUILD LINEAR REGRESSION MODEL

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully!")


# PREDICTION

y_pred = model.predict(X_test)


# MODEL EVALUATION

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np

# R² Score
r2 = r2_score(y_test, y_pred)

# MSE
mse = mean_squared_error(y_test, y_pred)

# RMSE
rmse = np.sqrt(mse)

print("MODEL EVALUATION - REGRESSION")

print("R² Score :", r2)
print("MSE      :", mse)
print("RMSE     :", rmse)

# CLASSIFICATION MODEL EVALUATION


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall   :", recall_score(y_test, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_test, y_pred, average="weighted"))


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)

linear_r2 = r2_score(y_test, linear_pred)
linear_mse = mean_squared_error(y_test, linear_pred)
linear_rmse = np.sqrt(linear_mse)

# Cross Validation
linear_cv = cross_val_score(
    linear_model, X, y, cv=5, scoring="r2"
)


ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_pred = ridge_model.predict(X_test)

ridge_r2 = r2_score(y_test, ridge_pred)
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)

ridge_cv = cross_val_score(
    ridge_model, X, y, cv=5, scoring="r2"
)


lasso_model = Lasso(alpha=1.0)
lasso_model.fit(X_train, y_train)

lasso_pred = lasso_model.predict(X_test)

lasso_r2 = r2_score(y_test, lasso_pred)
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)

lasso_cv = cross_val_score(
    lasso_model, X, y, cv=5, scoring="r2"
)


results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression"
    ],
    "R2 Score": [
        linear_r2,
        ridge_r2,
        lasso_r2
    ],
    "MSE": [
        linear_mse,
        ridge_mse,
        lasso_mse
    ],
    "RMSE": [
        linear_rmse,
        ridge_rmse,
        lasso_rmse
    ],
    "Cross Validation R2": [
        linear_cv.mean(),
        ridge_cv.mean(),
        lasso_cv.mean()
    ]
})

print("\nMODEL COMPARISON")
print(results)

print("\nCROSS VALIDATION SCORES")
print("Linear Regression:", linear_cv)
print("Ridge Regression :", ridge_cv)
print("Lasso Regression :", lasso_cv)