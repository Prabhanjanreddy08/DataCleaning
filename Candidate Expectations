1. Data Cleaning (Including Missing Values, Outliers, and Multi-collinearity)
Data cleaning is essential to ensure that the dataset is ready for building a robust fraud detection model. Below are the steps we take during data cleaning:

Handling Missing Values:
Imputation: For numerical columns with missing values, we use the SimpleImputer class from sklearn with the mean strategy, which replaces missing values with the mean of the respective column.
Handling Outliers:
Interquartile Range (IQR) Method: We use the IQR method to detect outliers. Outliers are defined as data points that fall outside of the range: 
Q1−1.5∗IQR,Q3+1.5∗IQR. These are then removed from the dataset.
Handling Multi-collinearity:
Correlation Matrix: We analyze the correlation matrix to detect high correlations between features. If the correlation exceeds a threshold (e.g., 0.9), one of the highly correlated variables is dropped to reduce multicollinearity.
Variance Inflation Factor (VIF): VIF can also be used to quantify the degree of multicollinearity. Features with high VIF (>5) might be removed.
        
Example Code for Data Cleaning:

          # Handling missing values
          imputer = SimpleImputer(strategy='mean')
          numeric_data_imputed = pd.DataFrame(imputer.fit_transform(numeric_data), columns=numeric_columns)
          
          # Handling outliers using IQR
          for column in numeric_columns:
              Q1 = data_imputed[column].quantile(0.25)
              Q3 = data_imputed[column].quantile(0.75)
              IQR = Q3 - Q1
              lower_bound = Q1 - 1.5 * IQR
              upper_bound = Q3 + 1.5 * IQR
              data_imputed = data_imputed[(data_imputed[column] >= lower_bound) & (data_imputed[column] <= upper_bound)]
          
          # Check for multicollinearity
          correlation_matrix = data_imputed.corr()

2. Fraud Detection Model Description
The model developed here is a Random Forest Classifier. Random Forest is a powerful ensemble learning method that is widely used in classification tasks, particularly when there is a need to handle complex datasets with many features.
Random Forest works by constructing a multitude of decision trees during training and outputting the class that is the mode of the classes (classification) of the individual trees.
Why Random Forest?
Non-linear Decision Boundaries: It can handle non-linear relationships.
Robustness to Overfitting: By averaging multiple trees, the model reduces the risk of overfitting.
Feature Importance: Random Forest can provide feature importance, which is useful for interpreting which features are more relevant in predicting fraud.

3. Variable Selection for the Model
In this project, variable selection is done through the following techniques:
Dropping High Cardinality Features: Features like nameOrig and nameDest, which represent unique identifiers, are dropped because they do not add value in predicting fraud.
Correlation Analysis: After removing high cardinality features, we analyze the correlation matrix to detect highly correlated features. Features with a correlation above 0.9 with other features are dropped.
Variance Thresholding: We use the VarianceThreshold method to remove features with low variance. Features with very low variance are unlikely to be informative and may add noise to the model.
Feature Importance: After training the Random Forest model, we analyze feature importance to understand which features are contributing the most to fraud detection.

4. Model Performance Evaluation
To evaluate the performance of the fraud detection model, we use the following metrics:
Classification Report: Includes precision, recall, F1-score, and support for both classes.
Confusion Matrix: Helps visualize the number of true positives, false positives, true negatives, and false negatives.
ROC-AUC Score: This metric is important for evaluating how well the model distinguishes between fraudulent and non-fraudulent transactions. A higher AUC indicates better performance.
Example Code for Model Evaluation:

          # Predictions
          y_pred = rf_model.predict(X_test)
          y_prob = rf_model.predict_proba(X_test)[:, 1]
          
          # Model evaluation
          print("Classification Report:\n", classification_report(y_test, y_pred))
          print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
          print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


5. Key Factors that Predict Fraudulent Transactions
The Random Forest model provides feature importance, which can help identify the key factors that predict fraudulent transactions. Some common factors might include:
Transaction Amount: High transaction amounts may be more likely to be flagged as fraudulent.
Transaction Time: Fraudulent activities may occur at unusual times or outside the typical behavior patterns of customers.
Frequency of Transactions: A sudden increase in transaction frequency might be a sign of fraud.
Location: If a transaction occurs from a location that is far from the customer’s usual location, it could indicate fraud.
Example Code for Feature Importance:

        # Feature importance
        importances = rf_model.feature_importances_
        feature_names = X.columns
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df)


6. Do These Factors Make Sense?
Yes, the identified factors generally make sense for fraud detection:
Transaction Amount: Fraudsters may try to conduct larger transactions to maximize their ill-gotten gains.
Transaction Time: Fraudulent transactions may happen at unusual times (e.g., late night or on holidays).
Frequency: A sudden spike in the number of transactions could indicate an attempt to quickly launder money.
Location: Fraud may involve transactions from locations that are geographically inconsistent with the customer's usual behavior.


7. Prevention Measures for the Company’s Infrastructure
Once the fraud detection system is in place, the company can implement the following prevention measures to minimize fraud risk:
Real-time Transaction Monitoring: Continuously monitor transactions for unusual patterns and flag suspicious activities in real-time.
Multi-factor Authentication: Implement multi-factor authentication for transactions, especially for high-value ones.
Anomaly Detection Systems: Integrate machine learning models that detect anomalies based on historical behavior to spot potential fraud.
Geo-fencing: Set up location-based alerts to block transactions from regions that are known to be high-risk or inconsistent with the customer's usual locations.       


8. How to Determine If the Prevention Measures Work?
To assess if the preventive measures work, the following actions can be taken:
Monitor Fraud Detection Metrics: Keep track of key performance indicators (KPIs) such as fraud detection rate, false positive rate, and the time taken to detect fraudulent transactions.
Conduct Periodic Audits: Regular audits of the model's predictions and its impact on fraud cases will help understand its effectiveness.
Customer Feedback: Analyze customer complaints related to false positives or undetected fraud.
A/B Testing: Conduct A/B testing by implementing the fraud detection measures for a subset of transactions and comparing fraud rates before and after implementation.
