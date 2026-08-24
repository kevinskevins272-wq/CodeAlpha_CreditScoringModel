import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("===================================")
print("       CREDIT SCORING MODEL")
print("===================================")

# 1. Load dataset
data = pd.read_csv("data/credit_data.csv")

print(f"\nTotal records: {len(data)}")

# 2. Separate features and target
X = data.drop("default", axis=1)
y = data["default"]

# 3. Create model
model = LogisticRegression(max_iter=2000)

# 4. Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5)

print("\nCross-Validation Results:")
print("Scores:", cv_scores)
print(f"Average Accuracy: {cv_scores.mean() * 100:.2f}%")

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Train model
model.fit(X_train, y_train)

# 7. Test model
y_pred = model.predict(X_test)

# 8. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# 9. Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 10. Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 11. Customer details
print("\n===================================")
print("       CUSTOMER DETAILS")
print("===================================")

age = int(input("Age: "))
income = float(input("Income: "))
credit_score = int(input("Credit Score: "))
loan_amount = float(input("Loan Amount: "))
employment_years = int(input("Employment Years: "))

# 12. Create customer data
customer = pd.DataFrame({
    "age": [age],
    "income": [income],
    "credit_score": [credit_score],
    "loan_amount": [loan_amount],
    "employment_years": [employment_years]
})

# 13. Make prediction
prediction = model.predict(customer)

# 14. Default probability
default_probability = model.predict_proba(customer)[0][1]

# 15. Credit risk score
risk_score = round((1 - default_probability) * 100, 2)

# 16. Risk level
if risk_score >= 80:
    risk_level = "LOW"
elif risk_score >= 50:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# 17. Display result
print("\n===================================")
print("       CREDIT RISK RESULT")
print("===================================")

if prediction[0] == 1:
    print("Prediction: High Risk - Possible Default")
else:
    print("Prediction: Low Risk - No Default")

print(f"Credit Risk Score: {risk_score}/100")
print(f"Risk Level: {risk_level}")
print(f"Probability of Default: {default_probability * 100:.2f}%")

print("===================================")