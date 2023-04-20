from strong_classifier import StrongClassifier, StrongClassifierChooser
import numpy as np
from sklearn.model_selection import train_test_split
class Architecture:
    """
    Build architecture of a cascade for a given:
    Ftarget: target false positive rate
    f: max false positive rate per layer
    d: 
    v_size: size of the validation set
    X: features of the training data (n_features, n_samples)
    y: labels of the training data (n_samples,)
    """
    def __init__(self, X, y, Ftarget=0.07, f=0.6, d=0.94, v_size=0.3, verbose=False):
        self.Ftarget = Ftarget
        self.f = f
        self.d = d
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X.T, y, test_size=v_size, stratify=y, random_state=42)
        self.X_train = self.X_train.T
        self.X_val = self.X_val.T


        self.v_size = v_size
        self.n_samples = X.shape[1]
        self.n_features = X.shape[0]

        self.strong_classifiers: list[StrongClassifier] = []
        self.verbose = verbose

    def build(self):
        """
        Build the architecture of the cascade
        """
        F0 = 1.0
        D0 = 1.0
        i = 0 # layer number        
        while F0 > self.Ftarget:
            i += 1
            if self.verbose:
                print(f"Building layer {i}")
            n_i = 0
            F1 = F0
            last_strong_classifier = None
            while F1 > self.f * F0:
                n_i += 1

                if self.verbose:
                    print(f"Building layer {i}, classifier {n_i}")

                strong_classifier_chooser = StrongClassifierChooser(self.X_train, self.y_train, n_i, verbose=self.verbose)
                strong_classifier = strong_classifier_chooser.train()
                last_strong_classifier = strong_classifier
                # change strong classifier to get confidence, update θ
                # sort by confidence
                # get accumulative detection rate 
                # find θ that exceeds d * D0

        

    
    