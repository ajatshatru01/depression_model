import pandas as pd
import numpy as np
#preprocessing and loading data
data = pd.read_csv('EEG.machinelearing_data_BRMH.csv')
print(data.head())
print("Data Description: \n", data.describe())
data.drop(["no.", "age", "eeg.date","education", "IQ", "sex"], axis=1, inplace =True)
data.rename(columns={"main.disorder":"main_disorder"}, inplace = True)
data.rename(columns={"specific.disorder":"specific_disorder"}, inplace = True)
print(data.head())
features_with_null=list(data.columns[data.isna().any()])
print(len(features_with_null))
main_disorders = list(data.main_disorder.unique())
print(main_disorders)
specific_disoders = list(data.specific_disorder.unique())
print(specific_disoders)
mood_data = data.loc[data['main_disorder'] == 'Mood disorder']
print(mood_data.head())
main_disorderstest = list(mood_data.main_disorder.unique())
print(main_disorderstest)
specific_mood_disoders = list(mood_data.specific_disorder.unique())
print(specific_mood_disoders)
depressive_data=data.loc[data['specific_disorder'] == 'Depressive disorder']
print(depressive_data.head())

from sklearn import preprocessing
pre_processing=preprocessing.LabelEncoder()
specific_disoders_encoding = pre_processing.fit_transform(mood_data["specific_disorder"])
features=["main_disorder" , "specific_disorder"]
mood_data.drop(features, axis=1, inplace=True)

print(mood_data.head())
features=mood_data.to_numpy()

# Target:
y = specific_disoders_encoding
#specify:
X = preprocessing.StandardScaler().fit_transform(features)

delta_cols = [col for col in mood_data.columns if 'delta' in col]
beta_cols = [col for col in mood_data.columns if 'beta' in col]
theta_cols = [col for col in mood_data.columns if 'theta' in col]
alpha_cols = [col for col in mood_data.columns if 'alpha' in col]

print(f"Number of Delta Columns : {len(delta_cols)}")
print(f"Number of Beta Columns : {len(beta_cols)}")
print(f"Number of Theta Columns : {len(theta_cols)}")
print(f"Number of Alpha Columns : {len(alpha_cols)}")

temp_features = delta_cols + beta_cols +theta_cols + alpha_cols
print(f"Number of items in temp_features : {len(temp_features)}")

req_features = mood_data[temp_features].to_numpy()
# the target
y = specific_disoders_encoding
#the features
X = preprocessing.StandardScaler().fit_transform(req_features)
# Train-Test Split & Model Training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

#evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))




 
