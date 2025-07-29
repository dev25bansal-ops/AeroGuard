# main.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------ Load Dataset ------------------
column_names = ['unit', 'cycle'] + \
               [f'op_set_{i}' for i in range(1, 4)] + \
               [f'sensor_{i}' for i in range(1, 22)]

df = pd.read_csv('train_FD001.txt', sep=' ', header=None)
df.drop([26, 27], axis=1, inplace=True)
df.columns = column_names

# ------------------ Calculate RUL ------------------
rul = df.groupby('unit')['cycle'].max().reset_index()
rul.columns = ['unit', 'max_cycle']
df = df.merge(rul, on='unit')
df['RUL'] = df['max_cycle'] - df['cycle']

# ------------------ Label Data ------------------
df['label'] = df['RUL'].apply(lambda x: 1 if x <= 30 else 0)

features = df.drop(['unit', 'cycle', 'max_cycle', 'RUL', 'label'], axis=1)
labels = df['label']

# ------------------ Preprocess ------------------
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------ Train Model ------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------ Visualize ------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ------------------ Realtime Sim ------------------
def predict_new_data(sensor_readings):
    sensor_readings = scaler.transform([sensor_readings])
    pred = model.predict(sensor_readings)
    return "⚠️ Fault Warning!" if pred[0] == 1 else "✅ Engine is Healthy."

# Test the function
print("\nTesting on sample input:")
print(predict_new_data(X_test[0]))
