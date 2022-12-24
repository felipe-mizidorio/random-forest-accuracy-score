import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

with open('data/census.csv', 'rb') as file:
    data_base = pd.read_csv(file)

x_census = data_base.iloc[:, 0:14].values
y_census = data_base.iloc[:, 14].values

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital_status = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital_status.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], remainder='passthrough')
x_census = onehotencoder_census.fit_transform(x_census).toarray()

scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

x_census_training, x_census_test, y_census_training, y_census_test = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

results = []
for number_of_trees in range(int(input('Number of trees: '))):
    random_forest_census = RandomForestClassifier(n_estimators=number_of_trees+1, criterion="entropy", random_state=0)
    random_forest_census.fit(x_census_training, y_census_training)
    predictions = random_forest_census.predict(x_census_test)
    results.append(accuracy_score(y_census_test, predictions)*100)

plt.plot(list(range(1, len(results)+1, 1)), results)
plt.xticks(range(1, len(results)+1))
plt.title('Accuracy of Random Forest')
plt.xlabel('Number of trees')
plt.ylabel('Accuracy(%)')
plt.show()