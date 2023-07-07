from classifier import BestClassifier, WeakClassifier, EQ
import numpy as np
import torch
import pickle as pkl
from typing import Dict, List
class StrongClassifier:
    """
    A strong classifier is a linear combination of weak classifiers

    weak_classifiers: list of weak classifiers
    alphas: list of weights of the weak classifiers

    θ: threshold of the strong classifier
    """
    def __init__(self, weak_classifiers: List[WeakClassifier], alphas: List[float]):
        self.weak_classifiers = weak_classifiers
        self.alphas = alphas
        self.θ = np.sum(self.alphas) / 2
        self.updatedIndecies = False
        self.p = 1
        self.n = 0

    def confidence(self, X: np.ndarray = None, f_idx_map: Dict[int, int] = None):
        """
        Calculate the confidence of the strong classifier
        input:
            X: data to predict, a numpy array of shape (n_features, n_samples)
            f_idx_map: f_idx_map[i] = j means that the i-th feature is the j-th feature in X
        output:
          confidence: confidence = (∑ αi * h(xi)), it will be compared with θ
        """

        confidence = np.zeros(X.shape[1])

        for i, weak_classifier in enumerate(self.weak_classifiers):
            confidence += self.alphas[i] * weak_classifier.predict(X, f_idx_map=f_idx_map)

        return confidence


    def predict(self, X: np.ndarray = None,
                f_idx_map: Dict[int, int] = None,):
        """
        Predict given data
        
        input:
            X: data to predict, a numpy array of shape (n_features, n_samples)
            f_idx_map: f_idx_map[i] = j means that the i-th feature is the j-th feature in X
        output:
          predictions: predictions
        """
            
        # predictions = np.zeros(X.shape[1]) # ∑ αi * h(xi)

        # for i, weak_classifier in enumerate(self.weak_classifiers):
        #     predictions += self.alphas[i] * weak_classifier.predict(X, f_idx_map=f_idx_map)
        
        # # self.θ = np.sum(self.alphas) / 2
        # # 1 if ∑ αi * h(xi) >= ∑ αi / 2 else 0
        # return predictions >= np.sum(self.alphas) / 2
        return self.confidence(X, f_idx_map=f_idx_map) >= self.θ
    
    def updateIndecies(self, f_idx_map: Dict[int, int]):
        """
        Update the indecies of features in each weak classifier
        """
        if self.updatedIndecies:
            return
        for weak_classifier in self.weak_classifiers:
            weak_classifier.updateIndecies(f_idx_map)
        self.updatedIndecies = True

    def confidence2(self, X: np.ndarray = None):
        if not self.updatedIndecies:
            raise Exception("Call updateIndecies first")
        
        confidence = np.zeros(X.shape[1])

        for i, weak_classifier in enumerate(self.weak_classifiers):
            confidence += self.alphas[i] * weak_classifier.predict2(X)

        return confidence
    
    def predict2(self, X: np.ndarray = None):
        """
        Predict given data
        call it only after updateIndecies

        input:
            X: data to predict, a numpy array of shape (n_features, n_samples)

        output:
            predictions: predictions
        """
        if not self.updatedIndecies:
            raise Exception("Call updateIndecies first")
        
        return self.confidence2(X) >= self.θ
    
    def changePN(self, p=1, n=0):
        """
        Change the positive and negative labels
        """
        self.p = p
        self.n = n
        for weak_classifier in self.weak_classifiers:
            weak_classifier.changePN(p, n)
            



    def save(self, filepath):
        with open(filepath, "wb") as f:
            pkl.dump(self, f)

class StrongClassifierChooser:
    def __init__(self, 
                 X: np.ndarray, 
                 y: np.ndarray, 
                 T: int,
                 batchsize: int = 1000,
                 verbose: bool = False,
                 delete_unused: bool = False,
                 equal_weights:bool=False):
        """
        X: training data, a numpy array of shape (n_features, n_samples)
        y: training labels, a numpy array of shape (n_samples,)
        T: number of iterations
        It is the layer of the cascade classifier

        TODO does self.X make it slower or more memory required?
        """
        self.X = X
        self.y = y
        self.T = T
        self.batchsize = batchsize
        self.weak_classifiers: list[WeakClassifier] = []
        self.alphas: list[float] = []
        self.n_samples = X.shape[1]
        self.n_features = X.shape[0]
        self.verbose = verbose
        self.delete_unused = delete_unused

        ones = y.sum()
        zeros = self.n_samples - ones
        # assert ones > 0 and zeros > 0, "No positive or negative samples"
        # if ones == 0:
        #     p_weight = 0
        #     n_weight = 1 / zeros
        # elif zeros == 0:
        #     p_weight = 1 / ones
        #     n_weight = 0
        # else:
        #     p_weight = 1 / (2 * ones)
        #     n_weight = 1 / (2 * zeros)
        if equal_weights:
            p_weight = 1 / self.n_samples
            n_weight = 1 / self.n_samples
        else:
            p_weight = 1 / (2 * ones) if ones > 0 else 0
            n_weight = 1 / (2 * zeros) if zeros > 0 else 0
        self.weights = np.where(y == 1, p_weight, n_weight) # if no negative samples, then all weights are 1 / (2 * ones)
        self.weights = self.weights / np.sum(self.weights)

        # assert EQ(np.sum(self.weights), 1), "Weights do not sum to 1" + " ∑w=" +str(np.sum(self.weights))


    def train(self, start_idx: int = 0):
        for t in range(start_idx, self.T):
            self.weights = self.weights / np.sum(self.weights)
            best_classifier = BestClassifier(self.X, self.y, self.weights, batchsize=self.batchsize, delete_unused=True, verbose=self.verbose)
            weak_classifier, _ = best_classifier.chooseClassifier()
            self.weak_classifiers.append(weak_classifier)
            ϵ = weak_classifier.ϵ + 1e-10 # avoid division by zero
            β = ϵ / (1 - ϵ)
            alpha = np.log(1 / β) # if β != 0 else 1000 # 1000 is a large number
            self.alphas.append(alpha)
            predictions = weak_classifier.predict(self.X)
            # α=ln(1/β) -> β = e^(-α) -> β^(1-e) = exp(-α * (1-e)) = exp(α * c), c: 1 if correct, 0 if incorrect
            self.weights = self.weights * np.exp(-alpha * (self.y == predictions))
            self.weights = self.weights / np.sum(self.weights)
            if self.verbose:
                print(f"Finished training weak classifier {t + 1} / {self.T}")
        

        if self.delete_unused:
            del self.X
            del self.y
            del self.weights

        self.strong_classifier = StrongClassifier(self.weak_classifiers, self.alphas)
        return self.strong_classifier

    def predict(self, X: np.ndarray = None, f_given=False):
        """
        Predict given data
        
        input:
            X: data to predict, a numpy array of shape (n_features, n_samples) or (n_samples,) if f_given is True
        output:
          predictions: predictions
        """

        # if X is None:
        #     X = self.X
            
        predictions = np.zeros(X.shape[0]) if f_given else np.zeros(X.shape[1])

        for i, weak_classifier in enumerate(self.weak_classifiers):
            predictions += self.alphas[i] * weak_classifier.predict(X, f_given=f_given)
        return predictions >= np.sum(self.alphas) / 2
    
    def save(self, path: str, type='pickle'):
        # if self.delete_unused:
        #     del self.X
        #     del self.y
        #     del self.weights
        tmpX = self.X
        tmpy = self.y
        tmpweights = self.weights
        self.X = None
        self.y = None
        self.weights = None

        if type == 'torch':
            torch.save(self, path)
        elif type == 'pickle':
            with open(path, "wb") as f:
                pkl.dump(self, f)
        self.X = tmpX
        self.y = tmpy
        self.weights = tmpweights

