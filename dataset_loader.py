# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from abc import ABC
from scipy.io import arff
import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
import sklearn.model_selection as model_selection
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataSetLoaderBaseClass(ABC):
    def __init__(self, file_path, randomness):
        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_y = None
        self.x = None
        self.y = None
        self.feature_names = None
        self.target_labels = None
        self.data = None
        self.dataset_name = None
        self.randomness = randomness
        self.file_path = file_path


class BreastCancerDataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path=None, randomness=1):
        super().__init__(file_path, randomness)
        dataset = load_breast_cancer()
        self.data = dataset
        self.dataset_name = 'Breast Cancer Data Set'
        self.file_path = file_path
        self.randomness = randomness
        self.feature_names = dataset.feature_names
        self.target_labels = dataset.target_names
        self.x = dataset.data
        self.y = dataset.target

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class SpamBaseDataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/spambase/spambase.data', randomness=1):
        super().__init__(file_path, randomness)
        dataset = pd.read_csv(file_path, header=None)
        self.data = dataset
        self.dataset_name = 'Spam Base Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.feature_names = ['word_freq_make',
                              'word_freq_address',
                              'word_freq_all',
                              'word_freq_3d',
                              'word_freq_our',
                              'word_freq_over',
                              'word_freq_remove',
                              'word_freq_internet',
                              'word_freq_order',
                              'word_freq_mail',
                              'word_freq_receive',
                              'word_freq_will',
                              'word_freq_people',
                              'word_freq_report',
                              'word_freq_addresses',
                              'word_freq_free',
                              'word_freq_business',
                              'word_freq_email',
                              'word_freq_you',
                              'word_freq_credit',
                              'word_freq_your',
                              'word_freq_font',
                              'word_freq_000',
                              'word_freq_money',
                              'word_freq_hp',
                              'word_freq_hpl',
                              'word_freq_george',
                              'word_freq_650',
                              'word_freq_lab',
                              'word_freq_labs',
                              'word_freq_telnet',
                              'word_freq_857',
                              'word_freq_data',
                              'word_freq_415',
                              'word_freq_85',
                              'word_freq_technology',
                              'word_freq_1999',
                              'word_freq_parts',
                              'word_freq_pm',
                              'word_freq_direct',
                              'word_freq_cs',
                              'word_freq_meeting',
                              'word_freq_original',
                              'word_freq_re',
                              'word_freq_edu',
                              'word_freq_table',
                              'word_freq_conference',
                              'char_freq_;',
                              'char_freq_(',
                              'char_freq_[',
                              'char_freq_!',
                              'char_freq_$',
                              'char_freq_#',
                              'capital_run_length_average',
                              'capital_run_length_longest']
        self.target_labels = ['Spam', 'NotSpam']

        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class ObesityDataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/ObesityDataSet/obesity_data_set.csv', randomness=1):
        super().__init__(file_path, randomness)
        dataset = pd.read_csv(file_path, header=None)
        self.data = dataset
        self.dataset_name = 'Obesity Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.feature_names = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC',
                              'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
        self.target_labels = ['Insufficient Weight', 'Normal Weight', 'Overweight Level I', 'Overweight Level II',
                              'Obesity Type I', 'Obesity Type II', 'Obesity Type III']

        # Pre Process data

        columns_to_encode = [0, 4, 5, 8, 9, 11, 14, 15, 16]
        label_encoder = preprocessing.LabelEncoder()

        df = self.data[columns_to_encode]
        df = df.apply(label_encoder.fit_transform)
        self.data = self.data.drop(columns_to_encode, axis=1)
        self.data = pd.concat([self.data, df], axis=1)

        # assign x and y
        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class RedWineDataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/wine/winequality-red.csv', randomness=1):
        super().__init__(file_path, randomness)
        dataset = pd.read_csv(file_path, delimiter=';', header=None)
        self.data = dataset
        self.dataset_name = 'Red Wine Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.feature_names = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                              "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
        self.target_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # assign x and y
        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])
        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class PhishingWebsite1DataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/phishing_website_1/data.arff', randomness=1):
        super().__init__(file_path, randomness)
        dataset = arff.loadarff(file_path)
        df = pd.DataFrame(dataset[0])
        attributes = pd.DataFrame(dataset[1])
        self.data = df
        self.dataset_name = 'Phishing Website Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.target_labels = attributes[-1:][0].to_list().pop()
        self.feature_names = attributes[0:-1][0].to_list()

        # assign x and y
        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.y = self.y.astype('int')

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class PhishingWebsite2DataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/phishing_website_2/data.arff', randomness=1):
        super().__init__(file_path, randomness)
        dataset = arff.loadarff(file_path)
        df = pd.DataFrame(dataset[0])
        attributes = pd.DataFrame(dataset[1])
        self.data = df
        self.dataset_name = 'Phishing Website 2 Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.target_labels = attributes[-1:][0].to_list().pop()
        self.feature_names = attributes[0:-1][0].to_list()

        # assign x and y
        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.y = self.y.astype('int')

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


class DiabetesDataSet(DataSetLoaderBaseClass):
    def __init__(self, split_value=0.2, file_path='datasets/diabetes/messidor_features.arff', randomness=1):
        super().__init__(file_path, randomness)
        dataset = arff.loadarff(file_path)
        df = pd.DataFrame(dataset[0])
        attributes = pd.DataFrame(dataset[1])
        self.data = df
        self.dataset_name = 'Diabetes Data Set'
        self.file_path = file_path
        self.randomness = randomness

        self.target_labels = attributes[-1:][0].to_list().pop()
        self.feature_names = attributes[0:-1][0].to_list()

        # assign x and y
        self.x = np.array(self.data.iloc[:, 0:-1])
        self.y = np.array(self.data.iloc[:, -1])

        # Standardizing data
        scaler = StandardScaler()
        self.x = scaler.fit_transform(self.x)
        self.y = self.y.astype('int')

        self.train_x, self.test_x, self.train_y, self.test_y = model_selection.train_test_split(
            self.x, self.y, test_size=split_value, random_state=self.randomness, stratify=self.y
        )


def load_datasets():
    # Use a breakpoint in the code line below to debug your script.
    # datasetSpamBase = SpamBaseDataSet()
    # datasetObesity = ObesityDataSet()
    # datasetWineQuality = RedWineDataSet()
    # datasetPhishingWebsite1 = PhishingWebsite1DataSet()
    datasetPhishingWebsite2 = PhishingWebsite2DataSet()
    datasetDiabetes = DiabetesDataSet()
    datasets = [datasetDiabetes, datasetPhishingWebsite2]
    return datasets
