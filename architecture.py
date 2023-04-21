from typing import List
from cascade import CascadeClassifier
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

        self.strong_classifiers: List[StrongClassifier] = []
        self.verbose = verbose

    def build(self):
        """
        Build the architecture of the cascade
        """
        F0 = 1.0
        D0 = 1.0
        i = 0 # layer number        
        cascaded_classifier = CascadeClassifier(self.X_train, self.y_train, verbose=self.verbose)
        
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
                # TODO check if use cascade_classifier or strong_classifier
                confidence = strong_classifier.confidence(self.X_val)
                requiredD = self.d * D0
                a_con = np.argsort(confidence)
                a_con = a_con[::-1]
                # get first index where sum of y[a_con[:i]] >= requiredD using accumulative sum
                y_acc = np.cumsum(self.y_val[a_con]) / np.sum(self.y_val)
                idx = np.argmax(y_acc >= requiredD) # first index where y_acc >= requiredD
                # make threshold average of confidence of idx and idx - 1
                # TODO check if this is correct
                if idx == 0:
                    threshold = confidence[a_con[idx]] - 0.01 # make sure it is smaller than the smallest confidence
                else:
                    threshold = (confidence[a_con[idx]] + confidence[a_con[idx - 1]]) / 2
                strong_classifier.θ = threshold
                # update F1: classified as positive / total number of negative samples
                F0 = np.sum(strong_classifier.predict(self.X_val) == 1) / np.sum(self.y_val == 0)
                if self.verbose:
                    print(f"False positive rate: {F0}")
                # update D0: classified as positive / total number of positive samples
                
                D0_temp = np.sum(strong_classifier.predict(self.X_val) == 1) / np.sum(self.y_val == 1)
                if D0_temp < D0 * self.d:
                    raise Exception("Detection rate is too low")
                D0 = D0_temp
                if self.verbose:
                    print(f"Detection rate: {D0}")
                last_strong_classifier = strong_classifier

            self.strong_classifiers.append(last_strong_classifier)
            cascaded_classifier.strong_classifiers = self.strong_classifiers
            # change X_train, y_train, X_val, y_val  to only include +ve samples and -ve samples that were misclassified
            if F1 > self.Ftarget:
                # get indices of misclassified samples
                # misclassified = np.where(self.y_train != last_strong_classifier.predict(self.X_train))[0]\
                # TODO check if this is correct or sould be last_strong_classifier.predict(self.X_val)?
                remaining_bool_train = self.y_train == 1 | (cascaded_classifier.predict(self.X_train) == 1 & self.y_train == 0) 
                remaining_bool_val = self.y_val == 1 | (cascaded_classifier.predict(self.X_val) == 1 & self.y_val == 0)
                self.X_train = self.X_train[:, remaining_bool_train]
                self.y_train = self.y_train[remaining_bool_train]
                self.X_val = self.X_val[:, remaining_bool_val]
                self.y_val = self.y_val[remaining_bool_val]

                
                
                # misclassified = np.where(self.y_val != cascaded_classifier.predict(self.X_val))[0] 
                # # get indices of +ve samples
                # positive = np.where(self.y_train == 1)[0]
                # # get indices of -ve samples
                # negative = np.where(self.y_train == 0)[0]
                # # get indices of -ve samples that were misclassified
                # negative_misclassified = np.intersect1d(misclassified, negative)
                # # get indices of +ve samples and -ve samples that were misclassified
                # misclassified = np.union1d(positive, negative_misclassified)
                # # update X_train, y_train, X_val, y_val
                # self.X_train = self.X_train[:, misclassified]
                # self.y_train = self.y_train[misclassified]
                # self.X_val = self.X_val[:, misclassified]
                # self.y_val = self.y_val[misclassified]
                # self.n_samples = self.X_train.shape[1]
                
            if self.verbose:
                print(f"Layer {i} built")


        if self.verbose:
            print("Architecture built")
            

        

    
    