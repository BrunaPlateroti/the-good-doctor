# Import relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay

# Import good doctor dataset
df = pd.read_csv('./good_doctor_data.csv') # 3840 rows x 13 columns
# Get the columns name
columns_name = list(df.columns)
# Rename the df columns to removes typos
df.rename(columns={'# Pregnancies': 'Pregnancies', 'Blood Chemestry~I': 'Blood_Chemistry_I',
                   'Blood Chemisty~II': 'Blood_Chemistry_II','Blood Chemisty~III': 'Blood_Chemistry_III',
                   'Blood Pressure': 'Blood_Pressure', 'Skin Thickness': 'Skin_Thickness',
                   'Genetic Predisposition Factor': 'Genetic_Predisposition_Factor',
                   "Air Qual'ty Index": 'Air_Quality_Index','$tate': 'State'}, inplace = True)
# Check for duplicates
#pd.set_option('display.max_columns', None)
print(df.duplicated().sum()) # 19 identical duplicates rows found
print(df['Unique_ID'].value_counts().head()) # Other 5 Unique_ID were found, tested 5 times each, however in this 5 times
# only Blood_Chemistry_III variable differ each time. I will keep this 'duplicates' because could be important for our
# analysis

# Show only the 19 duplicates that match exactly every columns
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

df = df.fillna(df.median(numeric_only=True)) # Impute with (fill) replace missing values with estimated or reasonable values.
# This prevents problem with machine learning. In this case we replace it with the median of the column.

### Exploratory data analysis
print(df['Outcome'].value_counts()) # 0 = 2489, 1 = 1332
print(df.describe()) # summary statistics

# Plot bar plot of Outcomes to check class imbalance - Our data is imbalanced
df['Outcome'].value_counts().plot(kind='bar')
plt.title('Class Distribution of Outcome')
plt.xlabel('Outcome')
plt.ylabel('Count')
plt.show()

# Bivariate analysis - Plot features vs Outcome to understand which one can be use to predict disease
df.boxplot(column='BMI', by='Outcome')
plt.title('BMI by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('BMI')
plt.show()

df.boxplot(column='Genetic_Predisposition_Factor', by='Outcome')
plt.title('Genetic_Predisposition_Factor by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('Genetic_Predisposition_Factor')
plt.show()

df.boxplot(column='Pregnancies', by='Outcome')
plt.title('Pregnancies by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('Pregnancies')
plt.show()

df.boxplot(column='Blood_Chemistry_I', by='Outcome')
plt.title('Blood_Chemistry_I by Outcome')
plt.suptitle('')
plt.xlabel('Outcome')
plt.ylabel('Blood_Chemistry_I')
plt.show()

# Correlation analysis with the Outcome
correlation_matrix = df.corr(numeric_only=True)
print(correlation_matrix['Outcome'].sort_values(ascending=False))
# Plot the matrix for better visualisation
plt.figure(figsize = (10,8))
plt.title('Correlation Matrix')
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.show()
# Red (strong correlation)- closer to +1 and blue (weak/no correlation)
# Looking at the last column 'Outcome'
# Blood Chemistry I is the strongest feature correlated to disease outcome (good for prediction)
# BMI and Pregnancies show moderate correlation with the outcome
# The other features can be considered as supporting features

## Machine Learning - Build a model for disease prediction

# Drop unique identifier (Unique_ID)
df = df.drop(columns = ['Unique_ID'])
# Set up the predictive values chosen from exploratory data analysis and correlation
feature = ['Blood_Chemistry_I', 'Pregnancies', 'BMI']
x = df[feature] # feature matrix - column you want to use for prediction
y = df['Outcome'] # target vector - label you want to predict
# Split the data - 20% testing and 80% training - random state is used for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify= y )
# Replace NaN values with the median from the training set, also for test set
x_train = x_train.fillna(x_train.median())
x_test = x_test.fillna(x_train.median())

# Build the model
rf_classifier = RandomForestClassifier(random_state = 42, class_weight = 'balanced') # class weight will help for imbalanced data
# Fit the model
rf_classifier.fit(x_train, y_train)

# Check feature importance after the model is fit - Check how much a single feature has contributed to the predictions
importances = rf_classifier.feature_importances_
pd.Series(importances, index=feature).plot(kind='bar')
plt.title("Feature Importances")
plt.ylabel("Importance")
plt.show()

# Make prediction
y_prediction = rf_classifier.predict(x_test) # Predicted label for the model
y_prediction_prob = rf_classifier.predict_proba(x_test)[:, 1]
# Calculate and plot confusion matrix
cm = confusion_matrix(y_test, y_prediction)
disp = ConfusionMatrixDisplay(confusion_matrix = cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print('Classification Report:')
print(classification_report(y_test, y_prediction))
print('ROC AUC Score', roc_auc_score(y_test, y_prediction_prob)) # Can be either 1 or 0.5
