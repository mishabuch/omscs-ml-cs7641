from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, GridSearchCV
import plots
import numpy as np


class Boosting:

    def __init__(self):
        # Initialize Classifier
        self.learner_name = 'AdaBoost'
        self.learner_model = None

    def run_experiment(self, dataset):
        # Fit regression model
        if dataset.dataset_name == 'Diabetes Data Set':
            regr_1 = tree.DecisionTreeClassifier(random_state=dataset.randomness,
                                                 ccp_alpha=0.01
                                                 )
            regr_2 = AdaBoostClassifier(regr_1)

            param_grid = {
                "base_estimator__splitter": ["best", "random"],
                "n_estimators": [50, 100, 1, 2, 10, 20, 30, 40]
            }

            regr_1.fit(dataset.train_x, dataset.train_y)
            regr_2.fit(dataset.train_x, dataset.train_y)

            # Predict
            y_1 = regr_1.predict(dataset.test_x)
            y_2 = regr_2.predict(dataset.test_x)

            # Plot the results
            print(f"Accuracy of the model regr_1 is {accuracy_score(y_1, dataset.test_y)}")
            print("Classification Report")
            print(classification_report(y_1, dataset.test_y))

            print(f"Accuracy of the model regr_2 is {accuracy_score(y_2, dataset.test_y)}")
            print("Classification Report")
            print(classification_report(y_2, dataset.test_y))

            # evaluate the model
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(regr_2, dataset.x, dataset.y, scoring='accuracy', cv=cv, n_jobs=-1,
                                       error_score='raise')

            print(f"Cross Validation Score is {np.mean(n_scores)}")

            # Grid Search
            # run grid search
            grid_search = GridSearchCV(regr_2, param_grid=param_grid, scoring='accuracy', cv=5)
            # execute the grid search
            grid_result = grid_search.fit(dataset.train_x, dataset.train_y)
            # summarize the best score and configuration
            print(f"Best parameters are {grid_result.best_params_}")

            predictions = grid_result.best_estimator_.predict(dataset.test_x)
            print(f"Accuracy of the model is {accuracy_score(predictions, dataset.test_y)}")
            print("Classification Report of Model")
            print(classification_report(predictions, dataset.test_y))

            learning_curve = plots.plot_learning_curve(grid_result.best_estimator_, "Learning Curves (Decision Trees)",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)

            learning_curve.show(block=False)
        else:
            regr_1 = tree.DecisionTreeClassifier(random_state=dataset.randomness,
                                                 ccp_alpha=0.00234
                                                 )
            regr_2 = AdaBoostClassifier(regr_1)

            param_grid = {
                "base_estimator__splitter": ["best", "random"],
                "n_estimators": [50, 100,150,200]
            }

            regr_1.fit(dataset.train_x, dataset.train_y)
            regr_2.fit(dataset.train_x, dataset.train_y)

            # Predict
            y_1 = regr_1.predict(dataset.test_x)
            y_2 = regr_2.predict(dataset.test_x)

            # Plot the results
            print(f"Accuracy of the model regr_1 is {accuracy_score(y_1, dataset.test_y)}")
            print("Classification Report")
            print(classification_report(y_1, dataset.test_y))

            print(f"Accuracy of the model regr_2 is {accuracy_score(y_2, dataset.test_y)}")
            print("Classification Report")
            print(classification_report(y_2, dataset.test_y))

            # evaluate the model
            cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
            n_scores = cross_val_score(regr_2, dataset.x, dataset.y, scoring='accuracy', cv=cv, n_jobs=-1,
                                       error_score='raise')

            print(f"Cross Validation Score is {np.mean(n_scores)}")

            # Grid Search
            # run grid search
            grid_search = GridSearchCV(regr_2, param_grid=param_grid, scoring='accuracy', cv=5)
            # execute the grid search
            grid_result = grid_search.fit(dataset.train_x, dataset.train_y)
            # summarize the best score and configuration
            print(f"Best parameters are {grid_result.best_params_}")

            predictions = grid_result.best_estimator_.predict(dataset.test_x)
            print(f"Accuracy of the model is {accuracy_score(predictions, dataset.test_y)}")
            print("Classification Report of Model")
            print(classification_report(predictions, dataset.test_y))

            learning_curve = plots.plot_learning_curve(grid_result.best_estimator_, "Learning Curves (Decision Trees)",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)

            learning_curve.show(block=False)
