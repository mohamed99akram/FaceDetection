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
    def __init__(self, X, y, Ftarget=0.07, f=0.6, d=0.94, v_size=0.3, verbose=False, maxperlayer=200):
        self.Ftarget = Ftarget
        self.f = f
        self.d = d
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X.T, y, test_size=v_size, stratify=y, random_state=42)
        self.X_train = self.X_train.T
        self.X_val = self.X_val.T


        self.v_size = v_size
        self.maxperlayer = maxperlayer

        self.strong_classifiers: List[StrongClassifier] = []
        self.verbose = verbose

    def build(self):
        """
        Build the architecture of the cascade
        """
        ## F0 = 1.0 
        F1 = 1.0 ##
        ## D0 = 1.0
        D1 = 1.0 ##
        i = 0 # layer number        
        cascaded_classifier = CascadeClassifier(self.X_train, self.y_train, verbose=self.verbose)
        
        ## while F0 > self.Ftarget:
        while F1 > self.Ftarget:##
            i += 1
            if self.verbose:
                print(f"Building layer {i}")
            n_i = 0
            ## F1 = F0
            F0 = F1 ##
            D0 = D1 ##
            last_strong_classifier = None
            strong_classifier_chooser = StrongClassifierChooser(self.X_train, self.y_train, n_i, verbose=self.verbose)
            while F1 > self.f * F0:
                n_i += 1

                if self.verbose:
                    print(f"Building layer {i}, classifier {n_i}")
                strong_classifier_chooser.T = n_i
                # observe that adding strong classifiers repeats job except for the last one, so we can save time by not repeating and starting from the last one
                strong_classifier = strong_classifier_chooser.train(n_i - 1)
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
                ## if idx == 0:
                ##     threshold = confidence[a_con[idx]] - 0.01 # make sure it is smaller than the smallest confidence
                if idx == len(a_con) - 1:
                    threshold = confidence[a_con[idx]] - 0.01
                else:
                    ## threshold = (confidence[a_con[idx]] + confidence[a_con[idx - 1]]) / 2
                    threshold = (confidence[a_con[idx]] + confidence[a_con[idx + 1]]) / 2
                    
                strong_classifier.θ = threshold
                # update F1: FP / total number of negative samples
                F1 = np.sum((strong_classifier.predict(self.X_val) == 1) & (self.y_val == 0)) / np.sum(self.y_val == 0)
                if self.verbose:
                    print(f"False positive rate: {F1}, required false positive rate: {self.f * F0}")
               
                # update D1: TP / total number of positive samples
                ## D0_temp = np.sum((strong_classifier.predict(self.X_val) == 1) & (self.y_val == 1)) / np.sum(self.y_val == 1)
                D1_temp = np.sum((strong_classifier.predict(self.X_val) == 1) & (self.y_val == 1)) / np.sum(self.y_val == 1)
                if self.verbose:
                    print(f"Detection rate: {D0}, required detection rate: {requiredD}")
                if D1_temp < D0 * self.d:
                    print(f"Detection rate is too low: {D1_temp}, required detection rate: {requiredD}, idx: {idx}")
                    raise Exception("Detection rate is too low")
                ## D0 = D0_temp
                D1 = D1_temp
                last_strong_classifier = strong_classifier
                if n_i >= self.maxperlayer:
                    break
            self.strong_classifiers.append(last_strong_classifier)
            cascaded_classifier.strong_classifiers = self.strong_classifiers
            
            # change X_train, y_train, X_val, y_val  to only include +ve samples and -ve samples that were misclassified
            if F1 > self.Ftarget:# and n_i < self.maxperlayer:
                # TODO check if this is correct or sould be last_strong_classifier.predict(self.X_val)?
                remaining_bool_train = (self.y_train == 1) | ((cascaded_classifier.predict(self.X_train) == 1) & (self.y_train == 0)) 
                remaining_bool_val = (self.y_val == 1) | ((cascaded_classifier.predict(self.X_val) == 1) & (self.y_val == 0))
                remaining_neg = np.sum((self.y_train == 0) & remaining_bool_train)
                if remaining_neg == 0:
                    print("No negative samples left")
                    break
                self.X_train = self.X_train[:, remaining_bool_train]
                self.y_train = self.y_train[remaining_bool_train]
                self.X_val = self.X_val[:, remaining_bool_val]
                self.y_val = self.y_val[remaining_bool_val]
                
            if self.verbose:
                print(f"Layer {i} built, false positive rate: {F0}, detection rate: {D0}, number of classifiers: {n_i}")


        if self.verbose:
            print("Architecture built")
            
        return cascaded_classifier

        

    
    