import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

class CrossValidator():

    '''
    El método inicializa la clase y recibe los siguientes parámetros
    dataset: Conjunto de datos del tipo objeto con dos elementos tipo ndarray {data:[], target:[]}
    folds: Número entero que define la cantidad de particiones para el dataset
    algorithm: Instancia de un clasificador de sklearn
    '''
    def __init__(self, dataset, folds, algorithm):
        self.dataset = dataset
        self.folds = folds
        self.algorithm = algorithm

    '''
    El método crea los grupos de entrenamiento del modelo en base a los folds definidos
    durante la creación de la clase, a su vez hace una mezcla de los datos para que la
    selección sea más aleatoria, como resultado retorna el conjunto de ítems de cada fold
    tanto de entrenamiento como validación.
    Ej: Fold 2: [[train][validation], [train][validation]]
    '''
    def get_folds(self):
        # join data + target
        full_dataset = np.column_stack((self.dataset.data, self.dataset.target))
        # shuffle dataset
        shuffle_dataset = np.random.permutation(full_dataset)
        # split dataset
        split_dataset = np.array_split(shuffle_dataset, self.folds)
        # separate in train & validation
        train = []
        validation = []
        for idx, sd in enumerate(split_dataset):
            v = sd
            t = np.row_stack([s for i, s in enumerate(split_dataset) if i != idx])
            train.append(t)
            validation.append(v)
        return train, validation
    
    '''
    Método que realiza la validación cruzada para cada uno de los folds para ello obtiene
    los respectivos grupos y posteriormente los itera para obtener los datos X,y de test
    y validación para el entrenamiento y comprobación del modelo. Finalmente el algoritmo
    devuelve la media y la desviación estándar de los scores obtenidos en cada iteración.
    '''
    def crossValidate(self):
        # fold_data = self.get_folds()
        scores = []
        train, validation = self.get_folds()
        # process cross validation iterations
        for idx, t in enumerate(train):
            # Separete target class in train data
            X_train = t[:, :-1]
            y_train = t[:, -1:].ravel()

            # Separete target class in validate data
            X_val = validation[idx][:, :-1]
            y_val = validation[idx][:, -1:].ravel()

            # Train model with train data
            self.algorithm.fit(X_train,y_train)

            # Get score with validate data
            score = self.algorithm.score(X_val, y_val)
            print(f"Iteration {idx + 1} with score : {score:.3f}")
            # Append score to get metrics
            scores.append(score)

        # Calculate metrics
        mean_score = np.mean(scores)
        sd_score = np.std(scores)

        return mean_score, sd_score


def main():
    rf_classifier = RandomForestClassifier()
    dt = datasets.load_iris()
    FOLDS = 9
    algorithm = rf_classifier
    x = CrossValidator(dt, FOLDS, algorithm)

    # print(x.get_folds())
    m,sd = x.crossValidate()
    print(f"Result of CrossValidation : Mean: {m:.2f}, STD: {sd:.2f}")

if __name__ == "__main__":
    main()