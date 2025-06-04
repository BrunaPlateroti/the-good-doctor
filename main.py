import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# Import dataset
df = pd.read_csv('./good_doctor_data.csv') # 3840 rows x 13 columns

# Get the column names
columns_name = list(df.columns)

# Rename columns to remove typos
df.rename(columns={'# Pregnancies': 'Pregnancies', 'Blood Chemestry~I': 'Blood_Chemistry_I',
                   'Blood Chemisty~II': 'Blood_Chemistry_II','Blood Chemisty~III': 'Blood_Chemistry_III',
                   'Blood Pressure': 'Blood_Pressure', 'Skin Thickness': 'Skin_Thickness',
                   'Genetic Predisposition Factor': 'Genetic_Predisposition_Factor',
                   "Air Qual'ty Index": 'Air_Quality_Index','$tate': 'State'}, inplace = True)

# Check for duplicates
print(df.duplicated().sum()) # 19 identical duplicates rows found

# Count Unique_ID
print(df['Unique_ID'].value_counts().head())

# Show only the 19 duplicates that match exactly every column
duplicates = df[df.duplicated(keep=False)]
print(duplicates)

# Drop the 19 identical duplicates - 3821 rows in total
df = df.drop_duplicates()

# Find missing values
print(df.isnull().sum())

# Replace missing values with NaN
cols_with_value_zero = ['Blood_Chemistry_I', 'Blood_Chemistry_II', 'Skin_Thickness']
for col in cols_with_value_zero:
    df[col] = df[col].replace(0, np.nan)

# Impute with (fill) replace missing values with estimated or reasonable values.
# This prevents problem with machine learning. In this case we replace it with the median of the column.
df = df.fillna(df.median(numeric_only=True))


### Exploratory data analysis
# Check class balance
print(df['Outcome'].value_counts()) # 0 = 2489, 1 = 1332

# Summary statistics to understand distribution
print(df.describe())

# Plot bar plot of Outcomes to check class balance - Our data is imbalanced
df['Outcome'].value_counts().plot(kind='bar')
plt.title('Class Distribution of Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.savefig("class_distribution.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot features vs Outcome to understand which one can be used to predict disease
# Bivariate analysis - BMI by Outcome
df.boxplot(column='BMI', by='Outcome')
plt.title('BMI by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('BMI')
plt.savefig("output/bmi_by_outcome.png", dpi=300, bbox_inches='tight')
plt.show()

# Bivariate analysis - Pregnancies by Outcome
df.boxplot(column='Pregnancies', by='Outcome')
plt.title('Pregnancies by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('Pregnancies')
plt.savefig("output/pregnancies_by_outcome.png", dpi=300, bbox_inches='tight')
plt.show()

# Bivariate analysis - Blood_Chemistry_I by Outcome
df.boxplot(column='Blood_Chemistry_I', by='Outcome')
plt.title('Blood_Chemistry_I by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('Blood_Chemistry_I')
plt.savefig("output/blood_chemistry_I_by_outcome.png", dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis with the Outcome
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix['Outcome'].sort_values(ascending=False))

# Plot the matrix for better visualisation - # Red (strong correlation)- closer to +1 and blue (weak/no correlation)
plt.figure(figsize = (10,8))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.savefig("output/correlation_matrix.png", dpi=300, bbox_inches='tight')
plt.show()


## Machine Learning - Build a model for disease prediction

# Drop unique identifier (Unique_ID)
df = df.drop(columns = ['Unique_ID'])

# Set up the predictive values chosen from exploratory data analysis and correlation
feature = ['Blood_Chemistry_I', 'Pregnancies', 'BMI']

# Feature matrix - column you want to use for prediction
x = df[feature]

# Target vector - label you want to predict
y = df['Outcome']

# Split the data - 20% testing and 80% training - random state is used for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify= y )

# Replace NaN values with the median from the training set, also for test set
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_train.median())

# Build the model
rf_classifier = RandomForestClassifier(random_state = 42, class_weight = 'balanced')

# Fit the model
rf_classifier.fit(x_train, y_train)

# Check feature importance after the model is fit - Check how much a single feature has contributed to the predictions
importances = rf_classifier.feature_importances_
pd.Series(importances, index=feature).plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.show()

# Make predictions
# Predicted label for the model
y_prediction = rf_classifier.predict(x_test)

# Get the predicted probabilities for each class from the Random Forest model.
# [:, 1] selects the probability for class 1 (disease) for each test sample.
y_prediction_prob = rf_classifier.predict_proba(x_test)[:, 1]

# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig("output/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

print('Classification Report:')
print(classification_report(y_test, y_prediction))
print('ROC AUC Score', roc_auc_score(y_test, y_prediction_prob)) # Can be either 1 or 0.5
