import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. LOAD AND PREPARE DATA
df = pd.read_csv('Churn_Modelling.csv')
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender']) # Female=0, Male=1

# One-Hot Encode Geography
df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# 2. TRAIN MODEL
X = df.drop('Exited', axis=1)
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("--- Customer Churn AI is Ready! ---")

# 3. INTERACTIVE PREDICTION LOOP
while True:
    print("\n" + "="*40)
    print("ENTER CUSTOMER DETAILS (or type '0' in Credit Score to exit)")
    
    try:
        score = int(input("Credit Score (e.g. 600): "))
        if score == 0: break
        
        age = int(input("Age: "))
        tenure = int(input("Tenure (Years with bank): "))
        balance = float(input("Account Balance: "))
        products = int(input("Number of Products (1-4): "))
        has_card = int(input("Has Credit Card? (1=Yes, 0=No): "))
        active = int(input("Is Active Member? (1=Yes, 0=No): "))
        salary = float(input("Estimated Annual Salary: "))
        gender = input("Gender (Male/Female): ").lower()
        geo = input("Country (France/Germany/Spain): ").lower()

        # --- DATA PROCESSING FOR INPUT ---
        # Convert Gender to 0 or 1
        gender_num = 1 if gender == "male" else 0
        
        # Convert Geography to match One-Hot Encoding (Germany, Spain)
        # In our model: France is 0,0 | Germany is 1,0 | Spain is 0,1
        geo_germany = 1 if geo == "germany" else 0
        geo_spain = 1 if geo == "spain" else 0

        # Combine into a single array
        input_data = [[score, gender_num, age, tenure, balance, products, has_card, active, salary, geo_germany, geo_spain]]
        
        # Scale the input just like the training data
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1] * 100

        print("-" * 40)
        if prediction[0] == 1:
            print(f"RESULT: ⚠️ HIGH RISK. Probability of leaving: {probability:.2f}%")
        else:
            print(f"RESULT: ✅ LOYAL CUSTOMER. Probability of leaving: {probability:.2f}%")
            
    except ValueError:
        print("Please enter valid numbers!")

print("Exiting...")