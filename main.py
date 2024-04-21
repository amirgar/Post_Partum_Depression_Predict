from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


''
path = r'data.csv'

"""
Age: 
40-45    364
35-40    349
30-35    338
45-50    271
25-30    181
"""

# Get data
data = pd.read_csv(path)

# Delete column
data.pop('Timestamp')


# Reformat data
replace_map_age = {'40-45': 1,
                   '35-40': 2,
                   '30-35': 3,
                   '45-50': 4,
                   '25-30': 5}

data = data.replace(to_replace='40-45', value=1)
data = data.replace(to_replace='35-40', value=2)
data = data.replace(to_replace='30-35', value=3)
data = data.replace(to_replace='45-50', value=4)
data = data.replace(to_replace='25-30', value=5)

replace_map_type_answer = {'Yes': 2,
                           'Sometimes': 1,
                           'No': 0,
                           'Two or more days a week': 1,
                           'Not at all': 1,
                           'Maybe': 1,
                           'Not interested to say': 1,
                           'Often': 1}

data = data.replace(to_replace='Yes', value=2)
data = data.replace(to_replace='Sometimes', value=1)
data = data.replace(to_replace='Two or more days a week', value=1)
data = data.replace(to_replace='Not at all', value=1)
data = data.replace(to_replace='Maybe', value=1)
data = data.replace(to_replace='Not interested to say', value=1)
data = data.replace(to_replace='No', value=0)
data = data.replace(to_replace='Often', value=1)

# Delete pollution in data
data = data[data['Irritable towards baby & partner'] != '<null>']
data = data[data['Feeling of guilt'] != '<null>']
data = data[data['Problems concentrating or making decision'] != '<null>']

'''
Convert type float -> int in data
data['Age'] = data['Age'].astype('int64')
data['Feeling sad or Tearful'] = data['Feeling sad or Tearful'].astype('int64')
data['Irritable towards baby & partner'] = data['Irritable towards baby & partner'].astype('int64')
data['Trouble sleeping at night'] = data['Trouble sleeping at night'].astype('int64')
data['Problems concentrating or making decision'] = data['Problems concentrating or making decision'].astype('int64')
data['Overeating or loss of appetite'] = data['Overeating or loss of appetite'].astype('int64')
data['Feeling anxious'] = data['Feeling anxious'].astype('int64')
data['Feeling of guilt'] = data['Feeling of guilt'].astype('int64')
data['Problems of bonding with baby'] = data['Problems of bonding with baby'].astype('int64')
data['Suicide attempt'] = data['Suicide attempt'].astype('int64')
'''

y = data['Feeling anxious']
data.pop('Feeling anxious')
x = data
x = x.fillna(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.19, shuffle=True)
rfc = RandomForestClassifier(random_state=95)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

'''
Model accuracy score with 10 decision-trees : 0.9860
Model accuracy score with Random Forest: 0.993006993006993
Model accuracy score with SVC: 0.8986013986013986
Model accuracy score with KNN: 0.8846153846153846
'''

''' 
MSE - 0.027972027972027972
RMSE - 0.16724840200141816
MAE - 0.055944055944055944
'''

if __name__ == '__main__':
    # Options of visualisation data
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    print(accuracy_score(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred, squared=False))
    print(mean_absolute_error(y_test, y_pred))
