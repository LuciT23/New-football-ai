import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Încarcare date prelucrate
df = pd.read_csv("meciuri_prelucrate_ai.csv")

# Păstrăm doar rândurile valide care au cote
df = df.dropna(subset=['Res', 'AvgCH', 'AvgCD', 'AvgCA']).copy()

# 2. Definire Target (Ce vrem să prezică AI-ul)
df['target_solist'] = df['Res'] # H = Gazde, D = Egal, A = Oaspeți
df['target_peste25'] = (df['HG'] + df['AG'] > 2.5).astype(int)

# 3. Selectare Caracteristici (Features)
features = [
    'forma_gazda_5m', 'forma_oaspete_5m',
    'g_marcate_gazda_medie', 'g_primite_gazda_medie',
    'g_marcate_oaspete_medie', 'g_primite_oaspete_medie',
    'AvgCH', 'AvgCD', 'AvgCA'
]

X = df[features]
y_solist = df['target_solist']
y_peste25 = df['target_peste25']

# 4. Împărțire date cronologic (Fără shuffle pentru a nu învăța din viitor)
X_train, X_test, y_train_solist, y_test_solist = train_test_split(
    X, y_solist, shuffle=False, test_size=0.2
)
_, _, y_train_p25, y_test_p25 = train_test_split(
    X, y_peste25, shuffle=False, test_size=0.2
)

# 5. Antrenare Model Solist (1X2)
model_solist = RandomForestClassifier(n_estimators=100, random_state=42)
model_solist.fit(X_train, y_train_solist)
acc_solist = accuracy_score(y_test_solist, model_solist.predict(X_test))

# 6. Antrenare Model Peste/Sub 2.5 Goluri
model_p25 = RandomForestClassifier(n_estimators=100, random_state=42)
model_p25.fit(X_train, y_train_p25)
acc_p25 = accuracy_score(y_test_p25, model_p25.predict(X_test))

print("="*40)
print(" REZULTATE ANTRENARE MODEL AI ")
print("="*40)
print(f"Acuratete Predictie Solist 1X2: {acc_solist * 100:.2f}%")
print(f"Acuratete Predictie Peste 2.5 Goluri: {acc_p25 * 100:.2f}%")
print("="*40)

# 7. Testare pe ultimul meci din baza de date
ultimul_meci = X.iloc[[-1]]
probabilitati = model_solist.predict_proba(ultimul_meci)[0]
prob_p25 = model_p25.predict_proba(ultimul_meci)[0][1]

clase = model_solist.classes_
print("\nExemplu de șanse calculate pentru un meci nou:")
for cls, prob in zip(clase, probabilitati):
    print(f" Șansă {cls}: {prob * 100:.1f}%")
print(f" Șansă Peste 2.5 Goluri: {prob_p25 * 100:.1f}%")
