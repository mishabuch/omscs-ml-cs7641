import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

import plots


class KNN:

    def __init__(self):
        # Initialize Classifier
        self.learner_name = 'K Nearest Neighbour Learner'
        self.learner_model = None

    def run_experiment(self, dataset):
        # KNN WITH CROSS VALIDATION
        if dataset.dataset_name == 'Diabetes Data Set':
            k_range = np.arange(40, 50)
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                knn = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                           metric='manhattan', n_jobs=4)
                knn.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = knn.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = knn.score(dataset.test_x, dataset.test_y)

            # Visualization of k values vs accuracy

            plt.title('k-NN: Varying Number of Neighbors')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()
            # create new a knn model
            knn2 = KNeighborsClassifier(n_jobs=4)
            leaf_range = [1, 2, 3, 4]
            # create a dictionary of all values we want to test for n_neighbors
            param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_range, 'weights': ['uniform', 'distance'],
                          'metric': ['manhattan', 'euclidean'],
                          'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
            # use gridsearch to test all values for n_neighbors
            knn_gscv = GridSearchCV(knn2, param_grid, cv=3, n_jobs=-1, verbose=1)
            # fit model to data
            knn_gscv.fit(dataset.train_x, dataset.train_y)
            print(f"Best parameters for this knn algorithm are {knn_gscv.best_params_}")
            print(f"Best score for this knn algorithm are {knn_gscv.best_score_}")

            # Cross validation of model
            # scores = cross_val_score(knn_gscv.best_estimator_, dataset.x, dataset.y, scoring='accuracy')
            # print(f"CV Scores mean GSV Search : {scores.mean()} ")
            knn_gscv.best_estimator_.predict(dataset.test_x)
            print(f"Knn GSCV Score {knn_gscv.best_estimator_.score(dataset.test_x, dataset.test_y)}")

            ## Learning Curve
            learning = plots.plot_learning_curve(knn_gscv.best_estimator_, "Learning Curves (KNN)", dataset.train_x,
                                      dataset.train_y,
                                      cv=5, n_jobs=4)

            learning.show(block=False)
            plt.show()

            ## PLOT TIMINGS
            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=dataset.randomness)
                start_time = time.time()

                knn_gscv.best_estimator_.fit(X_train, y_train)
                y_pred = knn_gscv.best_estimator_.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'KNN:Time Taken & Accuracy vs TestSet Sizes for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for KNN algorithms is {np.mean(time_taken)}")

        else:
            k_range = np.arange(6, 15)
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                knn = KNeighborsClassifier(n_neighbors=k, weights='uniform',
                                           metric='manhattan', n_jobs=4)
                knn.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = knn.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = knn.score(dataset.test_x, dataset.test_y)

            # Visualization of k values vs accuracy

            plt.title('k-NN: Varying Number of Neighbors')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()
            # create new a knn model
            knn2 = KNeighborsClassifier(n_jobs=4)
            # create a dictionary of all values we want to test for n_neighbors
            param_grid = {'n_neighbors': k_range, 'weights': ['uniform', 'distance'],
                          'metric': ['manhattan', 'euclidean']}
            # use gridsearch to test all values for n_neighbors
            knn_gscv = GridSearchCV(knn2, param_grid, cv=3, n_jobs=-1, verbose=1)
            # fit model to data
            knn_gscv.fit(dataset.train_x, dataset.train_y)
            print(f"Best parameters for this knn algorithm are {knn_gscv.best_params_}")
            print(f"Best score for this knn algorithm are {knn_gscv.best_score_}")

            # Cross validation of model
            scores = cross_val_score(knn_gscv.best_estimator_, dataset.x, dataset.y, scoring='accuracy')
            print(f"CV Scores mean GSV Search : {scores.mean()} ")
            knn_gscv.best_estimator_.predict(dataset.test_x)
            print(f"Knn GSCV Score {knn_gscv.best_estimator_.score(dataset.test_x, dataset.test_y)}")

            ## Learning Curve
            learning = plots.plot_learning_curve(knn_gscv.best_estimator_, "Learning Curves (KNN)", dataset.train_x,
                                      dataset.train_y,
                                      cv=5, n_jobs=4)

            learning.show(block=False)
            plt.show()

            ## PLOT TIMINGS
            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=dataset.randomness)
                start_time = time.time()

                knn_gscv.best_estimator_.fit(X_train, y_train)
                y_pred = knn_gscv.best_estimator_.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'KNN:Time Taken & Accuracy vs TestSet Sizes for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for KNN algorithms is {np.mean(time_taken)}")
