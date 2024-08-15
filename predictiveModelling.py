import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

records_count = int(input("Enter the number of records: "))

input_feature1 = []
input_feature2 = []
output_target = []

for _ in range(records_count):
    input_feature1.append(float(input("Enter value for InputFeature1: ")))
    input_feature2.append(float(input("Enter value for InputFeature2: ")))
    output_target.append(float(input("Enter value for OutputTarget: ")))

dataset_dict = {
    'InputFeature1': input_feature1,
    'InputFeature2': input_feature2,
    'OutputTarget': output_target
}

data_frame = pd.DataFrame(dataset_dict)

predictors = data_frame[['InputFeature1', 'InputFeature2']]
outcome = data_frame['OutputTarget']

train_predictors, test_predictors, train_outcome, test_outcome = train_test_split(predictors, outcome, test_size=0.2, random_state=42)

scaling_tool = StandardScaler()
train_predictors_scaled = scaling_tool.fit_transform(train_predictors)
test_predictors_scaled = scaling_tool.transform(test_predictors)

regression_model = LinearRegression()
regression_model.fit(train_predictors_scaled, train_outcome)

predicted_values = regression_model.predict(test_predictors_scaled)
error_value = mean_squared_error(test_outcome, predicted_values)

print('Mean Squared Error:', error_value)
print('Predictions:', predicted_values)
