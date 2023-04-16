from accumulative import BestClassifier, WeakClassifier
import numpy as np
import torch

class StrongClassifier:
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 T: int):
        """
        X: training data, a numpy array of shape (n_features, n_samples)
        y: training labels, a numpy array of shape (n_samples,)
        T: number of iterations
        It is the layer of the cascade classifier
        """
        self.X = X
        self.y = y
        self.T = T
        self.weak_classifiers: list[WeakClassifier] = []
        self.alphas: list[float] = []
        self.n_samples = X.shape[1]
        self.n_features = X.shape[0]

        ones = y.sum()
        zeros = self.n_samples - ones
        assert ones > 0 and zeros > 0, "No positive or negative samples"
        p_weight = 1 / (2 * ones)
        n_weight = 1 / (2 * zeros)
        self.weights = np.where(y == 1, p_weight, n_weight)
        assert np.sum(self.weights) == 1, "Weights do not sum to 1"

    def train(self):
        for t in range(self.T):
            self.weights = self.weights / np.sum(self.weights)
            best_classifier = BestClassifier(self.X, self.y, self.weights, batchsize=1000, delete_unused=True) # TODO: batchsize should be a parameter?
            weak_classifier, _ = best_classifier.chooseClassifier()
            self.weak_classifiers.append(weak_classifier)
            ϵ = weak_classifier.ϵ 
            β = ϵ / (1 - ϵ)
            alpha = np.log(1 / β)
            self.alphas.append(alpha)
            predictions = weak_classifier.predict(self.X)
            # α=ln(1/β) -> β = e^(-α) -> β^(1-e) = exp(-α * (1-e)) = exp(α * c), c: 1 if correct, 0 if incorrect
            self.weights = self.weights * np.exp(-alpha * (self.y == predictions))
            self.weights = self.weights / np.sum(self.weights)
            print(f"Finished training weak classifier {t + 1} / {self.T}")
            
    def predict(self, X: np.ndarray = None, f_given=False):
        """
        Predict given data
        
        input:
            X: data to predict, a numpy array of shape (n_features, n_samples) or (n_samples,) if f_given is True
        output:
          predictions: predictions
        """

        if X is None:
            X = self.X
            
        predictions = np.zeros(X.shape[0]) if f_given else np.zeros(X.shape[1])

        for i, weak_classifier in enumerate(self.weak_classifiers):
            predictions += self.alphas[i] * weak_classifier.predict(X, f_given=f_given)
        return predictions >= np.sum(self.alphas) / 2