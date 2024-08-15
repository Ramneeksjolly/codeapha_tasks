import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

num_records = int(input("Enter the number of records: "))

socio_economic_status = []
age = []
gender = []
saved = []

for _ in range(num_records):
    socio_economic_status.append(int(input("Enter Socio-Economic Status (1=Low, 2=Middle, 3=High): ")))
    age.append(int(input("Enter Age: ")))
    gender.append(int(input("Enter Gender (1=Male, 0=Female): ")))
    saved.append(int(input("Enter Saved (1=Yes, 0=No): ")))

data_dict = {
    'SES': socio_economic_status,
    'Age': age,
    'Gender': gender,
    'Outcome': saved
}

df_input = pd.DataFrame(data_dict)

features = df_input[['SES', 'Age', 'Gender']]
target = df_input['Outcome']

scaler_instance = StandardScaler()
features_scaled = scaler_instance.fit_transform(features)

logistic_model = LogisticRegression()
logistic_model.fit(features_scaled, target)

predictions = logistic_model.predict(features_scaled)

#print('Predicted Outcome:', predictions[0])
if(predictions[0]==1):
    print("you are safe!")
else:
    print("you are in danger.")    
