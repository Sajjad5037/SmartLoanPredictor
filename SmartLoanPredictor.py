import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox

# Step 1: Creating the DataFrame with loan approval data
data = {
    "Applicant ID": [1, 2, 3, 4, 5, 6],
    "Age": [25, 30, 22, 35, 28, 40],
    "Income": [50000, 60000, 45000, 80000, 55000, 90000],
    "Credit Score": [700, 800, 650, 750, 720, 780],
    "Employment Status": ["Employed", "Employed", "Unemployed", "Employed", "Employed", "Self-Employed"],
    "Loan Amount": [20000, 25000, 15000, 30000, 22000, 50000],
    "Approved": ["Yes", "Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)

# Step 2: Preprocessing the Data (Encoding and feature selection)
label_encoder = LabelEncoder()
df['Approved'] = label_encoder.fit_transform(df['Approved'])  # Encoding target variable

# Selecting features for training
X = df[['Age', 'Credit Score']]  # Features (Age, Credit Score)
y = df['Approved']  # Target variable (Loan approval status)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Model Training and Hyperparameter Tuning
clf = DecisionTreeClassifier(random_state=42)

# Defining a parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Performing GridSearchCV to tune the hyperparameters
grid_search = GridSearchCV(clf, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Getting the best model from the grid search
best_clf = grid_search.best_estimator_

# Step 4: Model Evaluation
accuracy = best_clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Visualizing the decision tree
plt.figure(figsize=(10, 6))
plot_tree(best_clf, filled=True, feature_names=['Age', 'Credit Score'], class_names=['No', 'Yes'])
plt.title("Decision Tree for Loan Approval")
plt.show()

# Step 5: Building the GUI for real-time predictions
def make_prediction():
    try:
        # Get user input from the entry fields
        age = int(entry_age.get())
        credit_score = int(entry_credit_score.get())

        # Prepare the data for prediction
        user_data = [[age, credit_score]]
        prediction = best_clf.predict(user_data)

        # Show prediction result in a message box
        if prediction[0] == 1:
            messagebox.showinfo("Prediction Result", "Loan Approved!")
        else:
            messagebox.showinfo("Prediction Result", "Loan Not Approved.")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values for age and credit score.")

# Creating the GUI window
root = tk.Tk()
root.title("Loan Approval Prediction")

# Age Label and Entry
tk.Label(root, text="Enter Age:").grid(row=0, column=0, padx=10, pady=5)
entry_age = tk.Entry(root)
entry_age.grid(row=0, column=1, padx=10, pady=5)

# Credit Score Label and Entry
tk.Label(root, text="Enter Credit Score:").grid(row=1, column=0, padx=10, pady=5)
entry_credit_score = tk.Entry(root)
entry_credit_score.grid(row=1, column=1, padx=10, pady=5)

# Predict Button
predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.grid(row=2, column=0, columnspan=2, pady=20)

# Run the GUI main loop
root.mainloop()
