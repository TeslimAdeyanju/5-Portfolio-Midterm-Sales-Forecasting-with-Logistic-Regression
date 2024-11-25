# SuperStore Sales Analysis and Prediction: A Data Science Approach

## Project Overview

This project showcases how data science techniques can be applied to analyze SuperStore’s sales data, uncover actionable insights, and predict future performance. Leveraging Python and its robust libraries, I explore key drivers of sales and profitability, develop predictive models, and provide recommendations to optimize revenue and profitability.

---

## Objective

The primary goal is to analyze sales data and build machine learning models to:
1. Predict **sales revenue** and **profit margins**.
2. Classify orders as profitable or non-profitable.
3. Provide data-driven strategies for improving business performance.

---

## Problem Statement

SuperStore’s sales data holds valuable insights that can drive strategic decisions. This project addresses the following:
1. **Identifying Financial Drivers**: Analyze customer segments, discounting practices, product categories, and regional performance to determine which factors influence revenue and profitability the most.
2. **Predictive Modeling**: Build machine learning models (linear regression and logistic regression) to forecast sales and classify profitability.
3. **Actionable Recommendations**: Provide practical suggestions for discount strategies, customer targeting, and inventory management.

---

## Dataset Overview

The dataset includes records of transactions with the following key features:

### Most Relevant Features
- **Order Date (`order_date`)**: Captures seasonality and sales trends.
- **Ship Date (`ship_date`)**: Indicates logistics efficiency and potential impact on customer satisfaction.
- **Product Name (`product_name`)**: Highlights products with high sales or profitability.
- **Segment (`segment`)**: Differentiates sales across customer segments.
- **Category (`category`)** and **Subcategory (`subcategory`)**: Group products for broader insights into sales patterns.
- **Region (`region`)**, **City (`city`)**, **State (`state`)**: Show geographical variations in sales performance.
- **Discount (`discount`)**: Directly affects sales volume and profitability.
- **Quantity (`quantity`)**: Strongly correlated with sales revenue.
- **Profit (`profit`)**: Key metric for profitability analysis.

### Less Relevant Features
- **Order ID (`order_id`)**: Unique identifier; not predictive of sales.
- **Zip Code (`zip`)**: Too granular; higher-level geographical features are more useful.

---

## Key Project Features

1. **Sales Prediction**: Linear regression is used to forecast sales revenue.
2. **Profitability Analysis**: Logistic regression is employed to classify orders as profitable or non-profitable.
3. **Actionable Insights**: Trends and patterns are extracted to guide strategic decisions.
4. **Recommendations**: Data-driven strategies for optimizing discounts, targeting customers, and managing inventory.

---

## Current Progress

### Logistic Regression Model
- **Objective**: Classify orders as profitable (1) or non-profitable (0).
- **Current Accuracy**:  
  Using validation data, the logistic regression model achieves an accuracy of **97.25%**:
  ```python
  print((y_val == profit_decision).mean())  # 0.9724862431215607


### Cross-Validation
- Implemented: K-Fold Cross-Validation with 10 folds for more robust evaluation.
- AUC Scores:
- Mean AUC across folds: 0.997 ± 0.001.

### Fine-Tuning
- Regularization Strength (C): Tuned using cross-validation:
- Optimal C Value: 0.5 with AUC of 0.999 ± 0.000.


### Findings

- Key Drivers:
- Discount: Significant impact on profitability and sales volume.
- Region and Customer Segment: Clear differences in sales performance based on geography and segment.
- Logistic Regression: Highly effective for classifying profitable orders with near-perfect AUC scores.
- Regularization: Fine-tuning C improved model performance.


