# ML Classification Assignment

## 1. Problem Statement

The objective of this project is to implement and compare multiple machine
learning classification algorithms on a common classification dataset.

The implemented classification models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest

All models were trained and evaluated on the same dataset. The models were
evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score and Matthews
Correlation Coefficient (MCC).

An interactive Streamlit application was also developed to allow users to
upload test data, select a classification model and view its evaluation
results.

---

## 2. Dataset Description

The Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine
Learning Repository was selected for this assignment.

The dataset contains 569 instances and 30 numerical features. It is a binary
classification problem where the target variable represents the diagnosis.

The target variable was encoded as:

- 0 = Benign
- 1 = Malignant

The dataset was divided into training and testing sets using an 80:20
stratified train-test split.

The dataset satisfies the assignment requirements of having at least
500 instances and at least 12 features.

---

## 3. GitHub Repository Link

Repository link:
https://github.com/Shruti-ML/ML-Classification-Assignment

---

## 4. Models Used

### 4.1 Logistic Regression

Logistic Regression was implemented as a linear classification model.
StandardScaler was applied before classification because the features have
different numerical scales.

### 4.2 Decision Tree Classifier

A Decision Tree Classifier was implemented using the training dataset.
The model learns a series of decision rules to classify the observations.

### 4.3 K-Nearest Neighbors (kNN)

K-Nearest Neighbors was implemented with k = 5. StandardScaler was applied
before classification because kNN is sensitive to feature scale.

### 4.4 Gaussian Naive Bayes

Gaussian Naive Bayes was used as the probabilistic classification model.
It assumes that the numerical features follow Gaussian distributions within
each class.

### 4.5 Random Forest

Random Forest was implemented as the ensemble classification model using
200 decision trees. The random forest combines multiple decision trees to
produce the final classification.

---

## 5. Evaluation Metrics

The following six evaluation metrics were calculated for every model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Accuracy

Accuracy represents the proportion of correctly classified observations
among all observations.

### AUC

AUC measures the ability of the classifier to distinguish between the two
classes across different classification thresholds.

### Precision

Precision measures the proportion of predicted positive observations that
are actually positive.

### Recall

Recall measures the proportion of actual positive observations correctly
identified by the classifier.

### F1 Score

F1 Score is the harmonic mean of precision and recall.

### MCC

Matthews Correlation Coefficient measures the quality of binary
classification while considering true and false positives and negatives.

---

## 6. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Random Forest | 0.9649 | 0.9942 | 1.0000 | 0.9048 | 0.9500 | 0.9258 |

---

## 7. Observations

### 7.1 Logistic Regression

Logistic Regression achieved an accuracy of 96.49% and the highest AUC
score of 0.9960 among all the evaluated models. It achieved a precision of
0.9750, recall of 0.9286 and F1 score of 0.9512. Its MCC score was 0.9245.
The model provided a strong and balanced classification performance and
obtained the highest average score of 0.9567.

### 7.2 Decision Tree

The Decision Tree achieved an accuracy of 92.98% and an AUC of 0.9246,
which were the lowest among the five models. Its precision, recall and F1
score were all 0.9048, while its MCC score was 0.8492. It therefore showed
the weakest overall performance among the evaluated models.

### 7.3 K-Nearest Neighbors

kNN achieved an accuracy of 95.61% and an AUC of 0.9823. It achieved a
precision of 0.9744, recall of 0.9048 and F1 score of 0.9383. Its MCC score
was 0.9058. Overall, kNN performed well but was slightly below Logistic
Regression and Random Forest.

### 7.4 Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 93.86% and an AUC of 0.9934.
It achieved perfect precision of 1.0000, but its recall was comparatively
lower at 0.8333. Its F1 score was 0.9091 and MCC was 0.8715. Although the
model achieved a high AUC, its lower recall reduced its overall performance.

### 7.5 Random Forest

Random Forest achieved an accuracy of 96.49%, equal to Logistic Regression.
It achieved perfect precision of 1.0000 and an MCC score of 0.9258, the
highest MCC among the evaluated models. Its recall was 0.9048 and F1 score
was 0.9500. Its overall average score of 0.9566 was only slightly lower than
Logistic Regression.

---

## 8. Overall Winner

Based on the comparison of Accuracy, AUC, Precision, Recall, F1 Score and
MCC, Logistic Regression showed the strongest overall performance.

Logistic Regression obtained the highest average score of 0.9567. It also
achieved the highest AUC, recall and F1 score among the evaluated models.
Random Forest was a very close second with an average score of 0.9566 and
achieved the highest precision and MCC.

Therefore, Logistic Regression was selected as the best-performing model for
this experiment.

---

## 9. Streamlit Application

The trained classification models were integrated into an interactive
Streamlit application.

The application provides the following features:

- CSV test-data upload
- Model selection dropdown
- Accuracy display
- AUC Score display
- Precision display
- Recall display
- F1 Score display
- MCC Score display
- Confusion Matrix
- Classification Report

### Live Streamlit Application

[ADD YOUR STREAMLIT LINK HERE]

---

## 10. How to Run the Application

Install the required dependencies using:

```bash
pip install -r requirements.txt
