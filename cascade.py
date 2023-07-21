import torch
import numpy as np

from classifier import BestClassifier, WeakClassifier
from strong_classifier import StrongClassifierChooser, StrongClassifier
import pickle as pkl
import joblib
import os
from typing import Dict
import glob
from tqdm import tqdm
import cv2
from copy import deepcopy

def _rep(n, x):
    return [x] * n

_default_layers = [2, 5, *_rep(3, 20), *_rep(2, 50),
                   *_rep(5, 100), *_rep(20, 200)]
class CascadeClassifier:
    def __init__(self, 
                 layers: list= _default_layers, 
                 batchsize: int = 1000,
                 verbose: bool = False):
        """
        layers: list of number of iterations for each layer
        batchsize: batchsize for training
        verbose: print training progress
        """
        self.layers = layers
        self.strong_classifiers: list[StrongClassifier] = []
        self.batchsize = batchsize
        self.verbose = verbose
        self.updatedIndecies = False

    def train(self,
              data,
              start = 0,
              equal_weights: bool = False,
              print_accuracy: bool = False,
              more_neg_dict=None,):
        """
        Train the cascade classifier
        ### input:
            data: object of class Data that contains X and y
            more_neg_path: path to directory of negative samples (get after first layer)
            equal_weights: if True, use equal weights for positive and negative samples
            print_accuracy: if True, print accuracy of each layer
            more_neg_dict: dictionary of arguments for getMoreNeg
        *Note* this function changes data.X, data.y, So, if you need them, copy before passing
        """
        for i in range(start, len(self.layers)):
            # +++++ Get more negatives +++++
            if more_neg_dict is not None and i > 0:
                remain_bool = ((data.y == 0) & (self.predict(data.X) == 1)) | (data.y == 1)
                data.X = data.X[:, remain_bool]
                data.y = data.y[remain_bool]

                zeros_cnt2 = np.sum(data.y == 0)

                # TODO make it better (this makes total negatives = req_cnt)
                req_cnt = more_neg_dict.get("req_cnt", 6000) 
                req_cnt = req_cnt - zeros_cnt2 # make all negatives are equal to req_cnt
                more_neg_dict["req_cnt"] = req_cnt
                
                # get more negative samples
                cnt_ret = self.getMoreNeg(data=data, **more_neg_dict)

                if self.verbose:
                    zeros3 = np.sum(data.y == 0)
                    print("Added", cnt_ret, "negative samples, Total negatives: ", zeros3)

            # +++++++ Train current layer +++++
            layer = self.layers[i]
            strong_classifier_chooser = StrongClassifierChooser(data.X, data.y, layer, batchsize=self.batchsize, verbose=False, equal_weights=equal_weights)
            strong_classifier = strong_classifier_chooser.train(layer_num = i + 1)
            self.strong_classifiers.append(strong_classifier)

            # ++++++ Print accuracy of current layer +++++
            if print_accuracy:
                predictions = self.predict(data.X)
                print('For layer', i + 1, 'accuracy is', np.sum(predictions == data.y) / data.X.shape[1])
                


    def getMoreNeg(self, 
                   data,
                   more_neg_path: str = None,
                   n_per_img: int = 6,
                   resize_factor: float = 1,
                   req_cnt: int = 6000,
                   by_confidence: bool = False,
                   by_size: bool = False,
                   face_dict: dict = None,):
        """
        Get more negative samples from more_neg_path
        negative samples are from non-faces directory that are classified as faces (false positives)
        ### input
            more_neg_path: path to directory of negative samples
            n_per_img: number of negative samples to get from each image   
            req_cnt: required number of negative samples
            by_confidence: if True, get negative samples by confidence (highest confidence: more chance)
            by_size: if True, get negative samples by size (largest size: more chance)
            face_dict: arguments for FaceDetectorFeatures
        ### output
            cnt: count of chosen negative samples
        """
        from detect_face import FaceDetectorFeatures
        face_detector = FaceDetectorFeatures(**face_dict)
        # shuffle files
        files = glob.glob(more_neg_path + "/*.png")
        np.random.shuffle(files)
        cnt = 0
        req_cnt_per_img = n_per_img
        for i in tqdm(range(len(files)), desc='Images', colour='yellow'):
            img = cv2.imread(files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)))
            features = face_detector.find_face_features(img, n_faces=req_cnt_per_img, by_confidence=by_confidence, by_size=by_size) 

            cnt += features.shape[1]
            data.X = np.concatenate((data.X, features), axis=1)
            data.y = np.concatenate((data.y, np.zeros(features.shape[1], dtype=int)))
            if features.shape[1] < req_cnt_per_img: # keep rest for next image
                req_cnt_per_img += req_cnt_per_img - features.shape[1]
            else:
                req_cnt_per_img = n_per_img
            if cnt >= req_cnt:
                break
        return cnt


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
        works only with predict2, confidence2
        TODO make it work with predict
        """
        for strong_classifier in self.strong_classifiers:
            strong_classifier.changePN(p, n)


    def save(self, filename):
        with open(filename, "wb") as f:
            pkl.dump(self, f)


    def save_joblib(self, filename):
        joblib.dump(self, filename)