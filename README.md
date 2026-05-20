# Predicting Recurrence of Differentiated Thyroid Cancer Using Deep Learning Models

## Project Overview
This project focuses on predicting the recurrence of Differentiated Thyroid Cancer (DTC) using a combination of Machine Learning (ML) and Deep Learning (DL) models. The study utilizes a UCI dataset containing clinicopathological features to evaluate the effectiveness of various models, including Random Forest (RF), XGBoost, K-Nearest Neighbors (KNN), and Deep Neural Networks (DNN).

## Objectives
The primary objectives of this study are:
- To estimate the effectiveness of Machine Learning models (RF, XGBoost, KNN) and a Deep Learning model (DNN) in predicting the recurrence of DTC.
- To investigate the predictive performance of these models in terms of accuracy, precision, recall, and F1 score.
- To perform hyperparameter tuning and compare the results of ML and DL models.
- To validate the potential application of these predictive models in clinical settings.

## Methodology
1. **Data Source:** The UCI dataset with clinicopathological features related to Differentiated Thyroid Cancer.
2. **Models Used:**
   - Random Forest (RF)
   - XGBoost
   - K-Nearest Neighbors (KNN)
   - Deep Neural Network (DNN)
3. **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score
4. **Techniques:** Hyperparameter tuning and comparative analysis to identify the most suitable model.

## Understanding the Code Workflow
### Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense
```

### Load the Dataset
The dataset is loaded from the UCI repository and prepared for analysis.
```python
df = pd.read_csv('thyroid_cancer_data.csv')
print(df.head())
```

### Exploratory Data Analysis (EDA)
- Visualize the distribution of features using histograms and box plots.
- Check for missing values and outliers.
```python
sns.countplot(df['recurrence'])
plt.show()
```

### Data Preprocessing
- Handle missing values.
- Normalize numerical features.
- Encode categorical variables.
```python
X = df.drop('recurrence', axis=1)
y = df['recurrence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Model Training
#### Random Forest
```python
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
```
#### XGBoost
```python
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
```
#### K-Nearest Neighbors
```python
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
```
#### Deep Neural Network
```python
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=10)
```

### Model Comparison
The performance of each model is compared based on Accuracy, Precision, Recall, and F1 Score.
```python
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
```

## Results and Analysis
- **Random Forest (RF) and XGBoost:**
  - Both models exhibited remarkable performance with a test accuracy of **97.40%**.
  - RF demonstrated strong performance in **accuracy** and **precision**, making it suitable for medical scenarios where false positives should be minimized.
  - XGBoost excelled in **recall**, making it ideal for scenarios requiring high true positive detection in complex cases.

- **Deep Neural Network (DNN):**
  - Achieved a test accuracy of **94.81%**.
  - Suitable for handling intricate feature interactions and non-linear data relationships.

- **K-Nearest Neighbors (KNN):**
  - Achieved a test accuracy of **93.51%**.
  - More suitable for small datasets, but less effective for complex detections such as DTC recurrence.

## Conclusion
The study demonstrates that both RF and XGBoost models outperform other models in predicting DTC recurrence. 
- **RF:** Ideal for applications prioritizing high precision to minimize false positives in medical diagnoses.
- **XGBoost:** Best suited for scenarios requiring high recall, ensuring no true positive cases are missed.
- **DNN:** A valuable tool for non-linear data analysis, though slightly less accurate than RF and XGBoost.
- **KNN:** Less suitable for complex medical prediction tasks but can be effective for small datasets.

These insights highlight the importance of selecting models based on specific prediction requirements, whether prioritizing accuracy, precision, or recall.

## Future Scope and Applications
The findings of this project indicate significant potential for RF and XGBoost models in healthcare applications, particularly in predicting disease recurrence, patient outcomes, and treatment effectiveness.

Future work can focus on:
- **Data Collection:** Gathering larger amounts of clinical patient data to improve model robustness.
- **Personalized Treatment:** Using predictive models to identify high-risk patients and enhance personalized treatment strategies.
- **Generative AI:** Exploring advanced generative AI models to improve interpretability and ensure healthcare providers can trust and understand the predictions.
- **Multi-Modal Data Integration:** Incorporating textual and imagery data to combine clinical records and medical imaging, improving prediction accuracy and robustness.

## Key Takeaways
- Random Forest is highly precise, minimizing false positives in medical diagnoses.
- XGBoost excels in recall, ensuring no true positive cases are missed.
- DNN handles intricate feature interactions effectively.
- KNN is more suitable for smaller datasets but less effective for complex scenarios.

This project underscores the importance of model selection in healthcare applications and highlights future opportunities to improve predictive performance in real-world clinical settings.
