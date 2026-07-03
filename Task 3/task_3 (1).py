import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 1. Load the Data
df = pd.read_csv('Task 3 and 4_Loan_Data.csv')

# Drop 'customer_id' as it has no predictive power
X = df.drop(columns=['customer_id', 'default'])
y = df['default']

# 2. Split the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Comparative Analysis: Train Multiple Models
# --- Model A: Logistic Regression ---
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_probs = lr_model.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)

# --- Model B: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)
rf_probs = rf_model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Logistic Regression AUC: {lr_auc:.4f}")
print(f"Random Forest AUC: {rf_auc:.4f}")
print("Using Logistic Regression for the final pricing model due to standard industry practice.\n")

# 4. Expected Loss Function
def calculate_expected_loss(
    credit_lines, 
    loan_amt, 
    total_debt, 
    income, 
    years_employed, 
    fico_score, 
    model=lr_model, 
    recovery_rate=0.10
):
  
    
    # Create a dataframe for the input properties matching the training features
    borrower_data = pd.DataFrame([{
        'credit_lines_outstanding': credit_lines,
        'loan_amt_outstanding': loan_amt,
        'total_debt_outstanding': total_debt,
        'income': income,
        'years_employed': years_employed,
        'fico_score': fico_score
    }])
    
    # 1. Predict Probability of Default (PD)
    pd_estimate = model.predict_proba(borrower_data)[0][1]
    
    # 2. Exposure at Default (EAD)
    ead = loan_amt
    
    # 3. Loss Given Default (LGD)
    lgd = 1.0 - recovery_rate
    
    # 4. Calculate Expected Loss
    expected_loss = pd_estimate * ead * lgd
    
    return expected_loss, pd_estimate

# ==========================================
# TEST SCRIPT: Calculating Expected Loss
# ==========================================

# Test Case 1: A risky borrower (Low FICO, High Debt)
loss_risky, pd_risky = calculate_expected_loss(
    credit_lines=5,
    loan_amt=8500.0,
    total_debt=35000.0,
    income=45000.0,
    years_employed=2,
    fico_score=520
)

# Test Case 2: A safe borrower (High FICO, Low Debt)
loss_safe, pd_safe = calculate_expected_loss(
    credit_lines=0,
    loan_amt=2000.0,
    total_debt=2500.0,
    income=95000.0,
    years_employed=8,
    fico_score=780
)

print(f"--- Risky Borrower ---")
print(f"Probability of Default: {pd_risky:.2%}")
print(f"Expected Loss: ${loss_risky:,.2f} (on a $8,500 loan)")

print(f"\n--- Safe Borrower ---")
print(f"Probability of Default: {pd_safe:.2%}")
print(f"Expected Loss: ${loss_safe:,.2f} (on a $2,000 loan)")