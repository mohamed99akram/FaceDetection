import torch
import numpy as np

from classifier import BestClassifier, WeakClassifier
from strong_classifier import StrongClassifierChooser, StrongClassifier
import pickle as pkl
import os
from typing import Dict
def _rep(n, x):
    return [x] * n

_default_layers = [2, 5, *_rep(3, 20), *_rep(2, 50),
                   *_rep(5, 100), *_rep(20, 200)]
class CascadeClassifier:
    def __init__(self, X: np.ndarray, 
                 y: np.ndarray, 
                 layers: list= _default_layers, 
                 batchsize: int = 1000,
                 verbose: bool = False,
                 use_stored: bool = True):
        """
        X: training data, a numpy array of shape (n_features, n_samples)
        y: training labels, a numpy array of shape (n_samples,)
        layers: list of number of iterations for each layer
        batchsize: batchsize for training
        verbose: print training progress
        use_stored: if True, continue training from last stored strong classifier
        """
        self.X = X
        self.y = y
        self.layers = layers
        self.strong_classifiers: list[StrongClassifier] = []
        self.n_samples = X.shape[1]
        self.n_features = X.shape[0]
        self.n_layers = len(layers)
        self.batchsize = batchsize
        self.verbose = verbose
        self.use_stored = use_stored
        self.updatedIndecies = False

    def train(self):
        """
        Train the cascade classifier
        return: training accuracy
        """
        start = 0
        # to continue training
        dirpath = "StrongClassifier/"
        lastSC = "StrongClassifier/lastSC.last"
        if not os.path.exists(dirpath):
            os.makedirs("StrongClassifier/")

        if os.path.exists(lastSC) and self.use_stored:
            with open(lastSC, "r") as f:
                start = int(f.read()) + 1
            for i in range(start):
                with open(dirpath+f"strong_classifier_{i}.pkl", "rb") as f:
                    self.strong_classifiers.append(pkl.load(f))
            print(f"Continue training from layer {start + 1} / {self.n_layers}")
        
        chosen_samples = np.ones(self.n_samples, dtype=bool)
        for i in range(start, self.n_layers):
            layer = self.layers[i]
            if self.verbose:
                print(f"$$$$$$$ Training layer {i + 1} / {self.n_layers} $$$$$$$")
            # strong_classifier_chooser = StrongClassifierChooser(self.X, self.y, layer, batchsize=self.batchsize, verbose=self.verbose)
            strong_classifier_chooser = StrongClassifierChooser(self.X[:, chosen_samples], self.y[chosen_samples], layer, batchsize=self.batchsize, verbose=self.verbose)
            strong_classifier = strong_classifier_chooser.train()
            self.strong_classifiers.append(strong_classifier)

            if self.verbose:
                print(f"$$$$$$$$$ Finished training layer {i + 1} / {self.n_layers} $$$$")
            
            strong_classifier.save(dirpath+f"strong_classifier_{i}.pkl")
            
            # write strong classifier number in lastSC.txt
            with open(lastSC, "w") as f:
                f.write(str(i))
                
            # Keep Positive Samples and Misclassified Negative Samples. rem_pfn: remaining positive and false positive
            # TODO self.predict? or strong_classifier.predict?
            tmp_bool = ((self.y[chosen_samples] == 0) & (self.predict(self.X[:, chosen_samples]) == 1)) | (self.y[chosen_samples] == 1)
            # chosen_samples = np.where(tmp_bool, True, False)
            chosen_samples[chosen_samples] = tmp_bool
            # now size of chosen_samples = 
            if self.verbose:
                print("Chosen samples:", chosen_samples.shape)
                print(f"%%%%%%% Layer {i + 1} / {self.n_layers} has remaining y=1: {np.sum(self.y[chosen_samples] == 1)}, y=0: {np.sum(self.y[chosen_samples] == 0)} %%%%%%%")

            # if no negative samples left, break
            if np.sum(self.y[chosen_samples] == 0) == 0:
                print("At layer", i + 1, "no negative samples left")
                # break

            # rem_pfp = (self.y == 0 & self.predict(self.X) == 1) | self.y == 1
            # self.X = self.X[:, rem_pfp]
            # self.y = self.y[rem_pfp]

            # # predict with current strong classifier
            # predictions = strong_classifier.predict(self.X)
            # # remove all samples that are correctly classified
            # self.X = self.X[:, predictions != self.y]
            # self.y = self.y[predictions != self.y]
            
        # return accuracy
        predictions = self.predict(self.X)
        return np.sum(predictions == self.y) / self.X.shape[1]


    def predict(self,
                X: np.ndarray, 
                f_idx_map: Dict[int, int] = None,):
        """
        Predict given data
        
        input:
            X: data to predict, a numpy array of shape (n_features, n_samples)
            f_idx_map: f_idx_map[i] = j means that the i-th feature is the j-th feature in X
        output:
          predictions: predictions

        TODO TEST THIS
        """

        predictions = np.ones(X.shape[1], dtype=bool)
        for strong_classifier in self.strong_classifiers:
            # if predictions is all false, break
            if not np.any(predictions):
                break
            if f_idx_map is not None:
                # get the features that are used in this strong classifier
                predictions[predictions] =\
                      predictions[predictions] & strong_classifier.predict(X[:,predictions], f_idx_map)
            else:
                predictions[predictions] =\
                      predictions[predictions] & strong_classifier.predict(X[:, predictions])
        return np.where(predictions, 1, 0)
    
    def confidence(self, X: np.ndarray, f_idx_map: Dict[int, int] = None):
        """
        Return confidence of each sample
        """
        confidences = np.zeros(X.shape[1])
        for strong_classifier in self.strong_classifiers:
            if f_idx_map is not None:
                confidences += strong_classifier.confidence(X, f_idx_map)
            else:
                confidences += strong_classifier.confidence(X)
        return confidences
    
    def updateIndecies(self, f_idx_map: Dict[int, int]):
        """
        Update the indecies of features in each weak classifier
        """
        if self.updatedIndecies:
            return
        for strong_classifier in self.strong_classifiers:
            strong_classifier.updateIndecies(f_idx_map)
        self.updatedIndecies = True

    def predict2(self, X: np.ndarray):
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


        predictions = np.ones(X.shape[1], dtype=bool)
        for strong_classifier in self.strong_classifiers:
            # if predictions is all false, break
            if not np.any(predictions):
                break
            predictions[predictions] =\
                  predictions[predictions] & strong_classifier.predict2(X[:, predictions])
        return np.where(predictions, 1, 0)
    
    def confidence2(self, X: np.ndarray):
        """
        Return confidence of each sample
        """
        confidences = np.zeros(X.shape[1])
        for strong_classifier in self.strong_classifiers:
            confidences += strong_classifier.confidence2(X)
        return confidences


    def updateThreshold(self, θ):
        for strong_classifier in self.strong_classifiers:
            strong_classifier.θ = θ

    def changePN(self, p=1, n=0):
        """
        Change positive and negative labels
        """
        for strong_classifier in self.strong_classifiers:
            strong_classifier.changePN(p, n)


    def save(self, filename):
        # save without self.X
        tmpX = self.X
        self.X = None
        with open(filename, "wb") as f:
            pkl.dump(self, f)
        self.X = tmpX