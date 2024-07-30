import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv(r"C:\Users\Admin\Downloads\Churn_Modelling.csv")
df.ffill(inplace=True)
label_encoder = LabelEncoder()
if 'Gender' in df.columns:
    df['Gender'] = label_encoder.fit_transform(df['Gender'])

if 'Geography' in df.columns:
    df['Geography'] = label_encoder.fit_transform(df['Geography'])
features = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
X = df[features]
y = df['Exited']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Not Churned', 'Churned'])
total_customers = len(df)
churned_customers = df['Exited'].sum()
not_churned_customers = total_customers - churned_customers
print(f'Total customers: {total_customers}')
print(f'Customers churned: {churned_customers}')
print(f'Customers not churned: {not_churned_customers}')
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
