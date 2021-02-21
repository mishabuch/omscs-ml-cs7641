import time

from sklearn import neural_network
from sklearn.metrics import accuracy_score, classification_report
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


class NeuralNetworks:

    def __init__(self):
        # Initialize Classifier
        self.learner_name = 'Neural Network'
        self.learner_model = None

    def run_experiment(self, dataset):
        if dataset.dataset_name == 'Diabetes Data Set':
            # Create Simple Neural Networks classifier
            self.learner_model = neural_network.MLPClassifier(activation='tanh', max_iter=1500)
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            predictions = self.learner_model.predict(dataset.test_x)
            print(f"Accuracy of the model is {accuracy_score(predictions, dataset.test_y)}")
            print("Classification Report of Model")
            print(classification_report(predictions, dataset.test_y))
            # self.plot_confusion_matrix(predictions, dataset.test_y, dataset.target_labels, dataset.dataset_name)

            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=42)
                start_time = time.time()
                mlp = neural_network.MLPClassifier(activation='tanh')
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)

            ## Plot Times taken by different models
            plt.title(f'NN: Accuracy & Time Taken VS TestData Size for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            # Create NNet specific to diabetes set
            # Test Various params and save accuracies
            activation = ['identity', 'logistic', 'relu', 'tanh']  # Best relu
            momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]  # Same as default
            learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]  # Same as default
            optimizer = ['sgd', 'adam', 'lbfgs']  # Best Adam
            train_accuracy = np.empty(len(activation))
            test_accuracy = np.empty(len(activation))
            time_taken = np.empty(len(activation))
            for i, k in enumerate(activation):
                start_time = time.time()
                mlp = neural_network.MLPClassifier(activation=k, max_iter=1500)
                mlp.fit(dataset.train_x, dataset.train_y)
                test_accuracy[i] = mlp.score(dataset.test_x, dataset.test_y)
                elapsed_time = time.time() - start_time
                train_accuracy[i] = mlp.score(dataset.train_x, dataset.train_y)
                time_taken[i] = elapsed_time

            print(f"Average time taken for NNet algorithms is {np.mean(time_taken)}")

            plt.title(f'Neural Network: Varying parameter activation for Dataset {dataset.dataset_name}')
            plt.plot(activation, test_accuracy, label='Testing Accuracy')
            plt.plot(activation, train_accuracy, label='Training Accuracy')
            plt.legend()
            plt.xlabel('activation')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            # create model with Keras to view loss and accuracy
            model = Sequential()
            model.add(Dense(100, input_dim=19, activation='tanh'))
            model.add(Dense(4, activation='tanh'))
            model.add(Dense(1, activation='tanh'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            history = model.fit(dataset.train_x, dataset.train_y, validation_split=0.3, epochs=200, batch_size=20,
                                verbose=0)
            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show(block=False)
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show(block=False)
            plt.show()
        else:
            # Create Simple Neural Networks classifier
            self.learner_model = neural_network.MLPClassifier(activation='relu',solver='adam',
                                                              max_iter=1500)
            self.learner_model.fit(dataset.train_x, dataset.train_y)
            predictions = self.learner_model.predict(dataset.test_x)
            print(f"Accuracy of the initial model is {accuracy_score(predictions, dataset.test_y)}")
            print("Classification Report of initial Model")
            print(classification_report(predictions, dataset.test_y))

            # Plot time taken for various sizes of database
            train_sizes = np.linspace(0.1, 0.9, 5)
            time_taken = np.empty(len(train_sizes))
            accuracy_scores = np.empty(len(train_sizes))
            for i, k in enumerate(train_sizes):
                X_train, X_test, y_train, y_test = train_test_split(
                    dataset.x, dataset.y, test_size=k, random_state=42)
                start_time = time.time()
                mlp = neural_network.MLPClassifier(activation='relu')
                mlp.fit(X_train, y_train)
                y_pred = mlp.predict(X_test)
                elapsed_time = time.time() - start_time
                time_taken[i] = elapsed_time
                accuracy_scores[i] = accuracy_score(y_pred, y_test)
            ## Plot Times taken by different models
            plt.title(f'NN:  Accuracy & Time Taken vs TestData sizes for {dataset.dataset_name}')
            plt.plot(train_sizes, time_taken, label='Time Taken Vs TestData Size')
            plt.plot(train_sizes, accuracy_scores, label='Accuracy Vs TestData Size')
            plt.legend()
            plt.xlabel('TestData Size')
            plt.ylabel('Time Taken')
            plt.show(block=False)
            plt.show()

            ## Plot Times taken by different models
            print(f"Average time taken for NNet algorithms is {np.mean(time_taken)}")

            # Create NNet specific to diabetes set
            # Test Various params and save accuracies
            activation = ['identity', 'logistic', 'relu', 'tanh']  # Best relu
            momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]  # Same as default
            learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]  # Same as default
            solver = ['sgd', 'adam', 'lbfgs']  # Best Adam
            train_accuracy = np.empty(len(activation))
            test_accuracy = np.empty(len(activation))
            for i, k in enumerate(activation):
                mlp = neural_network.MLPClassifier(activation=k)
                mlp.fit(dataset.train_x, dataset.train_y)
                train_accuracy[i] = mlp.score(dataset.train_x, dataset.train_y)
                test_accuracy[i] = mlp.score(dataset.test_x, dataset.test_y)
            plt.title(f'Neural Network: Varying parameter Activation for {dataset.dataset_name}')
            plt.plot(activation, test_accuracy, label='Testing Accuracy')
            plt.plot(activation, train_accuracy, label='Training Accuracy')
            plt.legend()
            plt.xlabel('Number of Neighbors')
            plt.ylabel('Accuracy')
            plt.show(block=False)
            plt.show()

            # create model
            model = Sequential()
            model.add(Dense(100, input_dim=9, activation='relu'))
            model.add(Dense(4, activation='relu'))
            model.add(Dense(1, activation='relu'))
            # Compile model
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            # Fit the model
            history = model.fit(dataset.train_x, dataset.train_y, validation_split=0.3, epochs=200, batch_size=20,
                                verbose=0)

            # summarize history for accuracy
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('Model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show(block=False)
            plt.show()

            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('Model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show(block=False)
            plt.show()