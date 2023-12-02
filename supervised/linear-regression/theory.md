# Linear Regression in Machine Learning

Linear Regression is a foundational technique in machine learning, often used for predicting numerical values based on previous data. It is a straightforward yet powerful tool for making predictions and understanding relationships between variables.

## Introduction

Linear Regression is a statistical method that models the relationship between a dependent variable and one or more independent variables. It assumes that the relationship between these variables is linear, meaning that a change in one variable results in a proportional change in another.

### Types of Linear Regression

1. **Simple Linear Regression**: Involves two variables - one independent and one dependent.
2. **Multiple Linear Regression**: Uses more than one independent variable.

## The Mathematics Behind Linear Regression

### Simple Linear Regression

The equation of a simple linear regression line is:

$y = mx + b$

Where:
- $y$ is the dependent variable (output).
- $x$ is the independent variable (input).
- $m$ is the slope of the line (how much $y$ changes for a unit change in $x$).
- $b$ is the y-intercept (value of $y$ when $x$ is 0).

### Multiple Linear Regression

The equation is a bit more complex:

$y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n$

Where:
- $b_0, b_1, ..., b_n$ are coefficients.
- $x_1, x_2, ..., x_n$ are independent variables.

## Cost Function and Gradient Descent

### Cost Function

The cost function, often represented as $J$, measures how off our predictions are from the actual values. The most common cost function in linear regression is the Mean Squared Error (MSE).

$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$

Where $\hat{y}_i$ is the predicted value.

### Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function. It iteratively adjusts the coefficients ($b_0, b_1, ..., b_n$) to find the best values that minimize the cost.

## Example: Predicting House Prices

Consider a simple example where we predict house prices based on their size (square footage).

### Data

| House Size (sq ft) | Price (1000s of $) |
|--------------------|-------------------|
| 850                | 300               |
| 900                | 320               |
| 1200               | 350               |
| 1500               | 400               |
| 2000               | 450               |

### Implementing Simple Linear Regression

1. **Plotting the Data**: Visualize the relationship between house size and price.
2. **Calculating the Coefficients**: Use the formula or a library like `scikit-learn` in Python.
3. **Making Predictions**: For a given house size, predict the price.

### Plotting the Regression Line

![Plot showing house size vs price with a fitted linear regression line](plot.png)

### Code Snippet

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Data
X = np.array([850, 900, 1200, 1500, 2000]).reshape(-1, 1)
y = np.array([300, 320, 350, 400, 450])

# Model
model = LinearRegression()
model.fit(X, y)

# Predicting
predicted_price = model.predict(np.array([[1250]]))
print("Predicted Price:", predicted_price)
```

## Conclusion

Linear Regression is a powerful yet simple tool for predictive modeling. Understanding its fundamentals is crucial for any aspiring machine learning practitioner. As models grow in complexity, the core principles of linear regression often remain applicable.