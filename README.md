## Assignment 1 

### Requirements:
- You would require Python > 3.6
- After cloning the project, please install the requirements by running this command "pip install -r requirements.txt"
- This would install all the dependencies and you are ready to run the project. 

#### You can find the Code here:
https://github.com/mishabuch/omscs-ml-cs7641

#### To run the project, you can run the main.py file. 
- Upon running the main.py file, you would be asked to choose a learner out of the five learners, for either of the two datasets. Please enter the number of your choice of learner and dataset. 

##### Learners
1. DecisionTree
2. KNN
3. NeuralNetworks
4. Boosting
5. SVM

##### Datasets

1. Diabetes Retinatherapy
2. Phishing Websites

#### The datasets for diabetes and phishing websites are in the datasets/diabetes and datasets/phishing_website_2 folders respectively.

- The UCI Machine Learning Repository links for both are here:
1. http://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set
2. http://archive.ics.uci.edu/ml/datasets/Website+Phishing

###### Note: I have tried to use non blocking plot functions through out the code, but there are times when one manually needs to close the plot window in order to continue execution.

#### REFERENCES

During the course of this assignment, i have taken help from various documents as well as blogs on the internet in order to learn more about the behaviour of various learners. Here is the list of all the links used:

1. https://towardsdatascience.com/how-to-find-decision-tree-depth-via-cross-validation-2bf143f0f3d6
2. http://chrisstrelioff.ws/sandbox/2015/06/25/decision_trees_in_python_again_cross_validation.html
3. https://www.ritchieng.com/machine-learning-cross-validation/ 
4. Official Scikit documentation for various learners
5. https://medium.com/towards-artificial-intelligence/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf
6. https://towardsdatascience.com/decision-tree-classifier-and-cost-computation-pruning-using-python-b93a0985ea77
7. https://stackoverflow.com/questions/31231499/pruning-and-boosting-in-decision-trees
8. https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
9. https://towardsdatascience.com/predicting-wine-quality-with-several-classification-techniques-179038ea6434

###### Just For Fun : I tried various other datasets as part of this assignment in order to select 2. All of them are available in the dataset_loader.py, ready and massaged to use. You you try them out by tweaking the code in main.py. 
