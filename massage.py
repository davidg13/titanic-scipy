import pandas
from sklearn.preprocessing import StandardScaler




class Massager:
    def __init__(self):
        self._scaler = StandardScaler()
        self.embark_code = {'S': 0, 'C': 1, 'Q': 2}
        self.title_code = {'Master': 0, 'Miss': 1, 'Ms': 2, 'Mr': 3, 'Mrs': 4,
                           'Mlle': 5, 'Mme': 6,
                           'Col': 7, 'Major': 8, 'Dr': 10, 'Rev': 11, 'Capt':9,
                           'Dona': 12, 'Don': 13, 'Lady':14, 'Sir':15,
                           'the Countess': 17, 'Jonkheer': 16}

    def transform(self, df, trainscale = False):
        features = df[['Name','Pclass','Sex','Age','SibSp','Parch','Embarked','Fare']]

        features['Age'] = features['Age'].astype(float)
        median_age = features['Age'].mean()
        features['Age'].fillna(median_age, inplace=True)

        mean_fare = features['Fare'].mean()
        features['Fare'].fillna(mean_fare,inplace=True)

        mode_plcass = features['Pclass'].mode()
        features['Pclass'].fillna(mode_plcass,inplace=True)

        mode_plcass = features['SibSp'].mode()
        features['SibSp'].fillna(mode_plcass,inplace=True)

        features['Male'] = features['Sex'].map(lambda x: 1 if x == 'male' else 0)
        features['EmbCode'] = features['Embarked'].map(lambda x: self.embark_code['S'] if pandas.isnull(x) else self.embark_code[x])

        features['Title'] = features['Name'].apply((lambda x : self.code_title(x)))
        features = features[['Title', 'Pclass', 'Age','SibSp','Male', 'Fare']]

        if trainscale:
            self._scaler.fit(features)

        features_array = features.as_matrix()
        features_array = self._scaler.transform(features_array)

        return features_array

    def code_title(self,name):
        spname = name.split(',')[1].strip()
        title = spname.split('.')[0]
        return self.title_code[title]
