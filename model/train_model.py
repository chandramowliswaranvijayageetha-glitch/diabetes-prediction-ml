import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../diabetes.csv')

df[['Glucose','BloodPressure','SkinThickness',
    'Insulin','BMI']] = df[['Glucose','BloodPressure',
    'SkinThickness','Insulin','BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

models = {
    'Random Forest':       RandomForestClassifier(n_estimators=100),
    'Logistic Regression': LogisticRegression(),
    'SVM':                 SVC(probability=True)
}

best_model, best_acc = None, 0
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name}: {acc*100:.2f}%")
    if acc > best_acc:
        best_acc, best_model = acc, model

joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler,     'scaler.pkl')
print("Model saved successfully!")
print(f"Best accuracy: {best_acc*100:.2f}%")

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
print("Heatmap saved!")