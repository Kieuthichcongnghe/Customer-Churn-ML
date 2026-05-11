import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

def pick(choices, weights, n):
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()
    return RNG.choice(choices, size=n, p=weights)

n = 500

ids            = [f"SYN-{i:06d}" for i in range(1, n + 1)]
gender         = pick(["Male", "Female"], [50, 50], n)
senior_citizen = pick([0, 1],             [85, 15], n)
partner        = pick(["Yes", "No"],      [48, 52], n)
dependents     = pick(["Yes", "No"],      [30, 70], n)
tenure         = RNG.integers(0, 13, size=n)   # 0–12 tháng

phone_service  = pick(["Yes", "No"], [90, 10], n)
multiple_lines = np.where(phone_service == "No", "No phone service",
                          pick(["Yes", "No"], [42, 58], n))

internet_service = pick(["DSL", "Fiber optic", "No"], [30, 60, 10], n)

def addon(yes_w):
    base = pick(["Yes", "No"], [yes_w, 100 - yes_w], n)
    return np.where(internet_service == "No", "No internet service", base)

online_security   = addon(25)
online_backup     = addon(45)
device_protection = addon(44)
tech_support      = addon(25)
streaming_tv      = addon(44)
streaming_movies  = addon(44)

contract         = pick(["Month-to-month", "One year", "Two year"], [80, 13, 7], n)
paperless_billing = pick(["Yes", "No"], [59, 41], n)
payment_method   = pick(
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"],
    [50, 20, 15, 15], n
)

base_charge = np.where(
    internet_service == "Fiber optic", RNG.uniform(70, 110, n),
    np.where(internet_service == "DSL", RNG.uniform(45, 75, n),
             RNG.uniform(18, 30, n))
)
addon_count = (
    (online_security   == "Yes").astype(int) +
    (online_backup     == "Yes").astype(int) +
    (device_protection == "Yes").astype(int) +
    (tech_support      == "Yes").astype(int) +
    (streaming_tv      == "Yes").astype(int) +
    (streaming_movies  == "Yes").astype(int)
)
monthly_charges = np.round(np.clip(base_charge + addon_count * 3.5 + RNG.normal(0, 2, n), 18.25, 118.75), 2)
total_charges   = np.where(tenure == 0, " ",
                           np.round(tenure * monthly_charges + RNG.normal(0, 30, n), 2).astype(str))

churn_prob = np.zeros(n)
churn_prob += np.where(tenure <= 12,                          0.25, 0)
churn_prob += np.where(internet_service == "Fiber optic",     0.10, 0)
churn_prob += np.where(contract == "Month-to-month",          0.15, 0)
churn_prob += np.where(payment_method == "Electronic check",  0.08, 0)
churn_prob += np.where(online_security == "No",               0.05, 0)
churn_prob += np.where(tech_support == "No",                  0.05, 0)
churn_prob += np.where(monthly_charges > 70,                  0.05, 0)
churn_prob -= np.where(contract == "Two year",                0.20, 0)
churn_prob -= np.where(tenure > 48,                           0.15, 0)
churn_prob  = np.clip(churn_prob, 0.03, 0.92)
churn       = np.where(RNG.random(n) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID":       ids,
    "gender":           gender,
    "SeniorCitizen":    senior_citizen,
    "Partner":          partner,
    "Dependents":       dependents,
    "tenure":           tenure,
    "PhoneService":     phone_service,
    "MultipleLines":    multiple_lines,
    "InternetService":  internet_service,
    "OnlineSecurity":   online_security,
    "OnlineBackup":     online_backup,
    "DeviceProtection": device_protection,
    "TechSupport":      tech_support,
    "StreamingTV":      streaming_tv,
    "StreamingMovies":  streaming_movies,
    "Contract":         contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod":    payment_method,
    "MonthlyCharges":   monthly_charges,
    "TotalCharges":     total_charges,
    "Churn":            churn,
})

df.to_csv("test.csv", index=False)
print(f"✓ Đã lưu test.csv — {len(df)} bản ghi | Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
