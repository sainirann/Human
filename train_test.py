from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


class HumanBotClassifier:

    def __init__(self):
        self.train_data = pd.read_csv(filepath_or_buffer='train_cleaned.csv')
        self.train_output = self.train_data['outcome']
        self.train_data.drop(['bidder_id', 'outcome'], axis=1, inplace=True)

        self.test_data = pd.read_csv(filepath_or_buffer='test_cleaned.csv')
        self.bidder_id = self.test_data['bidder_id']
        self.test_data.drop(['bidder_id'], axis=1, inplace=True)

    def feature_selection(self):
        s = SelectKBest(chi2, k=len(self.train_data.columns))
        s.fit_transform(self.train_data, self.train_output)
        columns = self.train_data.columns
        scores = s.scores_
        column_to_score = {columns[i]: scores[i] for i in range(len(scores))}
        column_to_score = {k: v for k, v in sorted(column_to_score.items(), key=lambda item: item[1], reverse=True)}
        top_30_features = [k for k in column_to_score.keys()][:30]
        print('Features Selected with scores:', [(k, column_to_score[k]) for k in top_30_features])
        print('\nFeatures Dropped: \n', set(column_to_score.keys()) - set(top_30_features), "\n")
        self.train_data.drop(self.train_data.columns.difference(top_30_features), axis=1, inplace=True)
        self.test_data.drop(self.test_data.columns.difference(top_30_features), axis=1, inplace=True)

    def grid_search_predict(self, classifier, grid_values, classifier_name) :
        grid_search_best = model_selection.GridSearchCV(classifier, grid_values, cv=4, scoring='roc_auc')
        test_output = pd.DataFrame(self.bidder_id)
        grid_search_best.fit(self.train_data, self.train_output)
        y_hat = grid_search_best.predict_proba(self.train_data)[:, 1]
        print(classifier_name, "Training Score: {}".format(roc_auc_score(self.train_output, y_hat)))
        predictions = [float(x[1]) for x in grid_search_best.predict_proba(self.test_data)]
        test_output['prediction'] = predictions
        return test_output

    def random_forest_classifier(self):
        grid_values = {'n_estimators': range(150, 320, 10), 'max_depth':  range(4, 8), 'max_features': range(3, 6)}
        test_output = self.grid_search_predict(RandomForestClassifier(), grid_values, "Random Forest Classifier")
        test_output.to_csv('random_forest_test_output.csv', mode='w', index=False)

    def logistic_regression(self):
        grid_values = {}
        test_output = self.grid_search_predict(LogisticRegression(solver='liblinear'), grid_values, "Logistic Regression Classifier")
        test_output.to_csv('logistic_regression_test_output.csv', mode='w', index=False)

    def ada_boost_classifier(self):
        grid_values = {'n_estimators': range(20, 70, 5)}
        test_output = self.grid_search_predict(AdaBoostClassifier(), grid_values, "AdaBoost Classifier")
        test_output.to_csv('ada_boost_test_output.csv', mode='w', index=False)

    def knn_classifier(self):
        grid_values = {'n_neighbors': np.arange(50, 70)}
        test_output = self.grid_search_predict(KNeighborsClassifier(), grid_values, "K Nearest Neighbour Classifier")
        test_output.to_csv('knn_test_output.csv', mode='w', index=False)


human_or_bot_classifier = HumanBotClassifier()
human_or_bot_classifier.feature_selection()
print("\nClassifiers \n")
human_or_bot_classifier.knn_classifier()
human_or_bot_classifier.logistic_regression()
human_or_bot_classifier.ada_boost_classifier()
human_or_bot_classifier.random_forest_classifier()
