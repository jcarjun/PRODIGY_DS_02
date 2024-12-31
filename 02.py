# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train_data = pd.read_csv(r"2nd_task\train.csv")
test_data = pd.read_csv(r"2nd_task\test.csv")
submission_template = pd.read_csv(r"2nd_task\gender_submission.csv")

# Step 1: Data Cleaning for Training Data
# Fill missing values in 'Age' with median
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Drop 'Cabin' column
train_data.drop(columns=['Cabin'], inplace=True)

# Fill missing values in 'Embarked' with mode
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

# Step 2: Data Cleaning for Test Data
# Fill missing values in 'Age' and 'Fare' with median
test_data['Age'].fillna(test_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(test_data['Fare'].median(), inplace=True)

# Drop 'Cabin' column
test_data.drop(columns=['Cabin'], inplace=True)

# Fill missing values in 'Embarked' with mode
test_data['Embarked'].fillna(test_data['Embarked'].mode()[0], inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
# Visualizations on training data
sns.countplot(data=train_data, x='Survived')
plt.title('Survival Count')
plt.show()

sns.countplot(data=train_data, x='Survived', hue='Sex')
plt.title('Survival by Gender')
plt.show()

# Add more visualizations as needed...

# Step 4: Train a Model (Logistic Regression as an example)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Select features and target for training
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})
train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

X = train_data[features]
y = train_data['Survived']

# One-hot encode categorical variables if necessary
X = pd.get_dummies(X, columns=['Pclass', 'Embarked'], drop_first=True)

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Validate the model
y_pred = model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))

# Step 5: Make Predictions on Test Data
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data['Embarked'] = test_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
X_test = test_data[features]
X_test = pd.get_dummies(X_test, columns=['Pclass', 'Embarked'], drop_first=True)

predictions = model.predict(X_test)

# Step 6: Prepare Submission File
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})

# Save to CSV
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'.")
