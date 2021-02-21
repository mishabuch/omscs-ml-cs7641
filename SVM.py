import time

from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import plots


class SVM:

    def __init__(self):
        # Initialize Classifier
        self.learner_name = 'Support Vector Machines'
        self.learner_model = None

    def run_experiment(self, dataset):
        # SVM
        if dataset.dataset_name == 'Diabetes Data Set':
            svm_simple = SVC()
            svm_simple.fit(dataset.train_x, dataset.train_y)
            y_pred = svm_simple.predict(dataset.test_x)
            print(classification_report(dataset.test_y, y_pred))
            # Create SVM classifier
            self.learner_model = SVC(C=15.0, kernel='linear', degree=3, gamma='scale',
                                     coef0=0.0, shrinking=True, probability=False,
                                     tol=1e-3, cache_size=200, class_weight=None,
                                     max_iter=-1, decision_function_shape='ovr',
                                     random_state=dataset.randomness
                                     )
            # Fit the classifier to the data
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            scores = cross_val_score(self.learner_model, dataset.x, dataset.y, cv=10)
            print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                                      scores.std()),
                  end="\n\n")
            predictions = self.learner_model.predict(dataset.test_x)
            print("Classification Report")
            print(classification_report(predictions, dataset.test_y))
            curve = plots.plot_learning_curve(self.learner_model, "SVM Learning Curve", dataset.x, dataset.y,
                                              cv=5, n_jobs=4)
            curve.show(block=False)
            plt.show()

            ## TRAINING/TEST ACCURACY SCORE
            k_range = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                svm = SVC(C=k, kernel='linear', degree=3, gamma='scale',
                          coef0=0.0, shrinking=True, probability=False,
                          tol=1e-3, cache_size=200, class_weight=None,
                          max_iter=-1, decision_function_shape='ovr',
                          random_state=dataset.randomness
                          )
                svm.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = svm.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = svm.score(dataset.test_x, dataset.test_y)

            # Visualization of k values vs accuracy

            plt.title('SVM: Varying C With Gamma as Scale and Kernel Linear')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            ## PLOT TIMINGS
            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=42)
                start_time = time.time()

                self.learner_model.fit(X_train, y_train)
                y_pred = self.learner_model.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'SVM: Varying DataSet Sizes vs Time Taken for Dataset {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for SVM algorithms is {np.mean(time_taken)}")

            # Using Grid Search CV
            param_grid = {'C': [15, 20, 25, 30, 35, 40, 50],
                          'gamma': ['scale', 'auto'],
                          'kernel': ['linear', 'rbf']}

            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=3)
            grid.fit(dataset.train_x, dataset.train_y)
            # print best parameter after tuning
            print(f" Best params for the dataset is {grid.best_params_}")
            # print how our model looks after hyper-parameter tuning
            print(grid.best_estimator_)
            grid_predictions = grid.best_estimator_.predict(dataset.test_x)

            # print classification report
            print(
                f"The classification report for the best estimator is {classification_report(dataset.test_y, grid_predictions)}")
            learning_curve = plots.plot_learning_curve(grid.best_estimator_, "SVM Learning Curve For best estimator",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)
            learning_curve.show(block=False)
            plt.show()
        else:

            svm_simple = SVC()
            svm_simple.fit(dataset.train_x, dataset.train_y)
            y_pred = svm_simple.predict(dataset.test_x)
            print(classification_report(dataset.test_y, y_pred))
            # Create SVM classifier
            self.learner_model = SVC(C=10.0, kernel='rbf', degree=3, gamma='scale',
                                     random_state=dataset.randomness)
            # Fit the classifier to the data
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            scores = cross_val_score(self.learner_model, dataset.x, dataset.y, cv=10)
            print("mean: {:.3f} (std: {:.3f})".format(scores.mean(),
                                                      scores.std()),
                  end="\n\n")
            predictions = self.learner_model.predict(dataset.test_x)
            print("Classification Report")
            print(classification_report(predictions, dataset.test_y))
            curve = plots.plot_learning_curve(self.learner_model, "SVM Learning Curve", dataset.x, dataset.y,
                                              cv=5, n_jobs=4)
            curve.show(block=False)
            plt.show()

            ## TRAINING/TEST ACCURACY SCORE
            k_range = np.linspace(5, 15, 10)
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                svm = SVC(C=k, kernel='rbf', degree=3, gamma='auto',
                          random_state=dataset.randomness
                          )
                svm.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = svm.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = svm.score(dataset.test_x, dataset.test_y)

            plt.title('SVM: Varying C With Gamma as Auto and Kernel RBF')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('C')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            ## PLOT TIMINGS
            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=42)
                start_time = time.time()

                self.learner_model.fit(X_train, y_train)
                y_pred = self.learner_model.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'SVM: Varying DataSet Sizes vs Time Taken for Dataset {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for SVM algorithms is {np.mean(time_taken)}")

            # Using Grid Search CV
            param_grid = {'C': [10, 10.5, 10.6, 10.7, 10.8, 10.9, 11],
                          'gamma': ['scale', 'auto'],
                          'kernel': ['linear', 'rbf']}

            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=3)
            grid.fit(dataset.train_x, dataset.train_y)
            # print best parameter after tuning
            print(f" Best params for the dataset is {grid.best_params_}")
            # print how our model looks after hyper-parameter tuning
            print(grid.best_estimator_)
            grid_predictions = grid.best_estimator_.predict(dataset.test_x)

            # print classification report
            print(
                f"The classification report for the best estimator is {classification_report(dataset.test_y, grid_predictions)}")
            learning_curve = plots.plot_learning_curve(grid.best_estimator_, "SVM Learning Curve For best estimator",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)
            learning_curve.show(block=False)
            plt.show()
