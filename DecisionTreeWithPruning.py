import time

from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import plots


class DecisionTree:

    def __init__(self):
        self.learner_name = 'Decision Tree Learner'
        self.learner_model = None

    def run_experiment(self, dataset):
        if dataset.dataset_name == 'Diabetes Data Set':
            # Decision Tree without Pruning
            self.learner_model = tree.DecisionTreeClassifier(random_state=dataset.randomness)
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            y_test_pred = self.learner_model.predict(dataset.test_x)
            print(f'Test score for simple tree {accuracy_score(y_test_pred, dataset.test_y)}')

            clf = self.learner_model
            # Decision Tree with Post-Pruning
            path = clf.cost_complexity_pruning_path(dataset.train_x, dataset.train_y)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            # print(ccp_alphas)
            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = tree.DecisionTreeClassifier(random_state=dataset.randomness, ccp_alpha=ccp_alpha)
                clf.fit(dataset.train_x, dataset.train_y)
                clfs.append(clf)

            clfs = clfs[:-1]
            train_acc = []
            test_acc = []
            for c in clfs:
                y_train_pred = c.predict(dataset.train_x)
                y_test_pred = c.predict(dataset.test_x)
                train_acc.append(accuracy_score(y_train_pred, dataset.train_y))
                test_acc.append(accuracy_score(y_test_pred, dataset.test_y))

            plt.title('Decision Trees: Varying CCP ALPHAS')
            plt.plot(ccp_alphas[:-1], test_acc, label='Testing Accuracy')
            plt.plot(ccp_alphas[:-1], train_acc, label='Training Accuracy')
            # plt.xticks(np.arange(min(ccp_alphas), max(ccp_alphas) + 1, 1.0))
            plt.legend()
            plt.xlabel('CCP Alpha Values')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            clf_ = tree.DecisionTreeClassifier(random_state=dataset.randomness,
                                               ccp_alpha=0.01)
            clf_.fit(dataset.train_x, dataset.train_y)
            y_test_pred = clf_.predict(dataset.test_x)

            print(f'Test score cost complexity pruning {accuracy_score(y_test_pred, dataset.test_y)}')

            ###### WHOLE NEW TEST
            param_grid = {"criterion": ["gini", "entropy"],
                          "min_samples_split": [6, 7, 8],
                          "max_depth": [5, 10, 15, 18],
                          "min_samples_leaf": [1, 2, 4],
                          "max_leaf_nodes": [26, 28, 29, 30, 32],
                          }

            dt = DecisionTreeClassifier()
            ts_gs = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
            ts_gs.fit(dataset.train_x, dataset.train_y)
            model = ts_gs.best_estimator_

            # test the returned best parameters
            print("\n\n-- Testing best parameters [Grid]...")
            print(ts_gs.best_params_)
            y_test_pred = model.predict(dataset.test_x)
            print(f'Test score {accuracy_score(y_test_pred, dataset.test_y)}')

            learning_curve = plots.plot_learning_curve(ts_gs.best_estimator_, "Learning Curves (Decision Trees)",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)

            learning_curve.show()

            k_range = np.arange(15, 25)
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                knn = DecisionTreeClassifier(criterion='entropy', max_depth=k, min_samples_leaf=2,
                                             min_samples_split=6, max_leaf_nodes=28)
                knn.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = knn.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = knn.score(dataset.test_x, dataset.test_y)

            # Visualization of k values vs accuracy

            print(f"Average accuracy for all algorithms {np.mean(test_accuracy)}")

            plt.title('Decision Trees: Varying max_depth')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('max_depth')
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
                    dataset.x, dataset.y, test_size=k, random_state=dataset.randomness)
                start_time = time.time()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'Decision Trees: Time Taken & Accuracy vs TestData Sizes for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for Decision Trees algorithms is {np.mean(time_taken)}")
        else:
            # Decision Tree without Pruning
            self.learner_model = tree.DecisionTreeClassifier(random_state=dataset.randomness)
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            y_test_pred = self.learner_model.predict(dataset.test_x)
            print(f'Test score for simple tree {accuracy_score(y_test_pred, dataset.test_y)}')

            clf = self.learner_model
            # Decision Tree with Post-Pruning
            path = clf.cost_complexity_pruning_path(dataset.train_x, dataset.train_y)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities
            # print(ccp_alphas)
            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = tree.DecisionTreeClassifier(random_state=dataset.randomness, ccp_alpha=ccp_alpha)
                clf.fit(dataset.train_x, dataset.train_y)
                clfs.append(clf)

            clfs = clfs[:-1]
            train_acc = []
            test_acc = []
            for c in clfs:
                y_train_pred = c.predict(dataset.train_x)
                y_test_pred = c.predict(dataset.test_x)
                train_acc.append(accuracy_score(y_train_pred, dataset.train_y))
                test_acc.append(accuracy_score(y_test_pred, dataset.test_y))

            plt.title('Decision Trees: Varying CCP ALPHAS')
            plt.plot(ccp_alphas[:-1], test_acc, label='Testing Accuracy')
            plt.plot(ccp_alphas[:-1], train_acc, label='Training Accuracy')
            # plt.xticks(np.arange(min(ccp_alphas), max(ccp_alphas) + 1, 1.0))
            plt.legend()
            plt.xlabel('CCP Alpha Values')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            clf_ = tree.DecisionTreeClassifier(random_state=dataset.randomness,
                                               ccp_alpha=0.00234)
            clf_.fit(dataset.train_x, dataset.train_y)
            y_test_pred = clf_.predict(dataset.test_x)

            print(f'Test score cost complexity pruning {accuracy_score(y_test_pred, dataset.test_y)}')

            ###### WHOLE NEW TEST
            param_grid = {"criterion": ["gini", "entropy"],
                          "min_samples_split": [2, 3, 4],
                          "max_depth": [5, 6, 7, 8, 10, 12, 14],
                          "min_samples_leaf": [2, 3, 4, 6, 8],
                          "max_leaf_nodes": [30, 32, 34, 36, 38, 40, 42, 44],
                          }

            dt = DecisionTreeClassifier()
            ts_gs = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)
            ts_gs.fit(dataset.train_x, dataset.train_y)
            model = ts_gs.best_estimator_

            # test the returned best parameters
            print("\n\n-- Testing best parameters [Grid]...")
            print(ts_gs.best_params_)
            y_test_pred = model.predict(dataset.test_x)
            print(f'Test score {accuracy_score(y_test_pred, dataset.test_y)}')

            learning_curve = plots.plot_learning_curve(ts_gs.best_estimator_, "Learning Curves (Decision Trees)",
                                                       dataset.x, dataset.y,
                                                       cv=5, n_jobs=4)

            learning_curve.show()

            k_range = np.arange(5, 15)
            train_accuracy = np.empty(len(k_range))
            test_accuracy = np.empty(len(k_range))
            for i, k in enumerate(k_range):
                knn = DecisionTreeClassifier(criterion='entropy', max_depth=k, min_samples_leaf=2,
                                             min_samples_split=6, max_leaf_nodes=28)
                knn.fit(dataset.train_x, dataset.train_y)
                # Compute accuracy on the training set
                train_accuracy[i] = knn.score(dataset.train_x, dataset.train_y)

                # Compute accuracy on the testing set
                test_accuracy[i] = knn.score(dataset.test_x, dataset.test_y)

            # Visualization of k values vs accuracy

            print(f"Average accuracy for all algorithms {np.mean(test_accuracy)}")

            plt.title('Decision Trees: Varying max_depth')
            plt.plot(k_range, test_accuracy, label='Testing Accuracy')
            plt.plot(k_range, train_accuracy, label='Training Accuracy')
            plt.xticks(np.arange(min(k_range), max(k_range) + 1, 1.0))
            plt.legend()
            plt.xlabel('max_depth')
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
                    dataset.x, dataset.y, test_size=k, random_state=dataset.randomness)
                start_time = time.time()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'Decision Trees:Time Taken & Accuracy vs TestData Size for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            print(f"Average time taken for SVM algorithms is {np.mean(time_taken)}")
