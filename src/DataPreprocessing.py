import pandas as pd
from sklearn.feature_extraction import DictVectorizer

class preprocess:

    def __init__(self):
        self.dv = DictVectorizer(sparse=False)
        self.age_mean = 0
        self.embarked_mode = 0

    def read_file(self, trainpath, testpath):
        cols = pd.read_csv(trainpath, nrows=1, sep=",").columns
        feature_names = cols.tolist()
        train_data = pd.read_csv(trainpath, sep=',')
        test_data = pd.read_csv(testpath, sep=',')
        # add the label to the end of train and pop passenger id

        train_data['Survived'] = train_data.pop('Survived')

        # # remove passenger id from featurenames and move label survived to the last
        feature_names = feature_names[2:] + [feature_names[1]]
        return feature_names, train_data, test_data

    def feature_processing(self, data, train=False):
        # perform feature selection and also update the features which are not properly modelled

        # remove string part from ticket and keep only numeric value
        ticket = data['Ticket']
        new_ticket = []
        for x in ticket:
            try:
                new_ticket.append(x.split(" ")[1])
            except:
                new_ticket.append(x.split(" ")[0])
        data['Ticket'] = new_ticket

        age = data['Age']
        data['Age'] = (data['Age'] - age.mean()) / (age.max() - age.min())
        fare = data['Fare']
        data['Fare'] = (data['Fare'] - fare.mean()) / (fare.max() - fare.min())
        # use prefix in name as feature
        # prefix_name = data['Name']
        # prefix = [x.split(",")[1].split(".")[0] for x in prefix_name]
        # data['Prefix_name'] = prefix

        # use family name from passenger name
        surname = data['Name']
        family_name = [x.split(",")[0] for x in surname]
        data['Family_name'] = family_name

        # make labels last feature in train
        # if 'Survived' in data.columns:
        #     data['Survived'] = data.pop('Survived')

        # apply replace empty values with mean value for numeric feature and mode for non numeric feature


        data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())
        data['Age'] = data['Age'].fillna(data['Age'].mean())

        # cabin is a very sparse feature so replace empty values with cabinless
        data['Cabin'] = data['Cabin'].fillna('cabinless')

        # ticket fare equivaent to Pclass
        data.pop('Ticket')

        # just use one feat family instead of Parch and SibSp
        # data['Family'] = data["Parch"] + data["SibSp"]
        # data['Family'].loc[data['Family'] > 0] = 1
        # data['Family'].loc[data['Family'] == 0] = 0
        # data.pop('Parch')
        # data.pop('SibSp')

        # perform one hot ecoding for categorical features
        if train:
            data.pop('PassengerId')
            survived = data.pop('Survived')
            data.pop('Name')
            # prefix_name = data.pop('Prefix_name')
            surname = data.pop('Family_name')
            cabin = data.pop('Cabin')
            pclass = data.pop('Pclass')
            data = self.dv.fit_transform(data.convert_objects(convert_numeric=True).to_dict(orient='records'))
            data = pd.DataFrame(data=data, columns=self.dv.feature_names_)
            # data['Name'] = name
            # data['Prefix_name'] = prefix_name
            data['Family_name']= surname
            data['Cabin'] = cabin
            data['Pclass'] = pclass
            data['Survived'] = survived
        else:
            p_id = data.pop('PassengerId')
            data.pop('Name')
            # prefix_name = data.pop('Prefix_name')
            surname = data.pop('Family_name')
            cabin = data.pop('Cabin')
            pclass = data.pop('Pclass')
            data = self.dv.transform(data.convert_objects(convert_numeric=True).to_dict(orient='records'))
            data = pd.DataFrame(data=data, columns=self.dv.feature_names_)
            data.insert(0, 'PassengerId', p_id)
            #data['Name'] = name
            # data['Prefix_name'] = prefix_name
            data['Family_name'] = surname
            data['Cabin'] = cabin
            data['Pclass'] = pclass

        dummies = pd.get_dummies(data['Pclass']).rename(columns=lambda x: '%s=' % ('Pclass') + str(x))
        data = pd.concat([data, dummies], axis=1)
        data = data.drop(['Pclass'], axis=1)
        data = data.drop(['Embarked'], axis=1)

        if train:
            data['Survived'] = data.pop('Survived')
        return data
