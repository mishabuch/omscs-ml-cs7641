# Press the green button in the gutter to run the script.
import dataset_loader
from Boosting import Boosting
from DecisionTreeWithPruning import DecisionTree
from KNN import KNN
from NeuralNetworks import NeuralNetworks
from SVM import SVM

if __name__ == '__main__':
    print("Please select the number against the learner you want to build the model for datasets:")
    print("1 -> Decision Tree")
    print("2 -> K-NN")
    print("3 -> Neural Networks")
    print("4 -> Boosting")
    print("5 -> SVM")

    switcher = {
        1: DecisionTree(),
        2: KNN(),
        3: NeuralNetworks(),
        4: Boosting(),
        5: SVM(),
    }
    datasets = dataset_loader.load_datasets()
    model_number = input("Enter your value: ")

    model = switcher.get(int(model_number), "Invalid learner")

    print("Please select the dataset:")
    print("1 -> Diabetes Retinatherapy")
    print("2 -> Phishing Website")
    dataset_number = input("Enter your value: ")
    if int(dataset_number) == 1:
        item = datasets[0]
    else:
        item = datasets[1]
    print(f"Dataset : {item.dataset_name}")
    print(f"Learner Selected : {model.learner_name}")
    model.run_experiment(item)
