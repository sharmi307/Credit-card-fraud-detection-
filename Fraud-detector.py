
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Example transaction dataset
data = pd.DataFrame({
    "amount": [
        100, 200, 50, 500, 10000, 15000,
        80, 120, 20000, 75, 300, 25000,
        60, 9000, 150, 18000, 400, 30000
    ],
    "transactions_per_day": [
        2, 3, 1, 4, 15, 20, 2, 1, 25,
        1, 3, 30, 1, 12, 2, 22, 3, 35
    ],
    "fraud": [
        0, 0, 0, 0, 1, 1, 0, 0, 1,
        0, 0, 1, 0, 1, 0, 1, 0, 1
    ]
})

X = data[["amount", "transactions_per_day"]]
y = data["fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Credit Card Fraud Detection")
print(classification_report(
    y_test, predictions, zero_division=0
))

def check_transaction(amount, transactions_per_day):
    transaction = pd.DataFrame([{
        "amount": amount,
        "transactions_per_day": transactions_per_day
    }])

    result = model.predict(transaction)[0]

    if result == 1:
        print("⚠️ Suspicious transaction: FRAUD")
    else:
        print("✅ Transaction classified as LEGITIMATE")

check_transaction(20000, 25)
