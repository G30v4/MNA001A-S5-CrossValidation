from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

class CrossValidator():

    def __init__(self, dataset, folds, algorithm):
        self.dataset = dataset
        self.folds = folds
        self.algorithm = algorithm

    def get_folds(self):
        # dataset.shuffle
        validation = []
        test = []
        return validation, test

    def crossValidate(self):
        scores = []
        mean_score = 0
        sd_score = 0       
        
        return mean_score, sd_score


rf_classifier = RandomForestClassifier()
dt = datasets.load_iris
folds = 2
algorithm = rf_classifier
x = CrossValidation(dt, folds, algorithm)

print(x.get_folds())