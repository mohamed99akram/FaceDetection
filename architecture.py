from typing import List
from cascade import CascadeClassifier
from strong_classifier import StrongClassifier, StrongClassifierChooser
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
import glob
from tqdm import tqdm

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
    def __init__(self, 
                 X, y, 
                 Ftarget=0.07, f=0.6, d=0.94, v_size=0.3, 
                 verbose=False, 
                 maxperlayer=200, maxlayers=10, 
                 batchsize=1000, delete_unused=False, equal_weights=False,
                 *args, **kwargs):
        self.Ftarget = Ftarget
        self.f = f
        self.d = d
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X.T, y, test_size=v_size, stratify=y, random_state=42)
        self.X_train = self.X_train.T
        self.X_val = self.X_val.T


        self.v_size = v_size
        self.maxperlayer = maxperlayer
        self.maxlayers = maxlayers
        self.batchsize = batchsize
        self.delete_unused = delete_unused
        self.equal_weights = equal_weights

        

        self.strong_classifiers: List[StrongClassifier] = []
        self.verbose = verbose

    def build(self,
              more_neg_path: str = None,
              *args, **kwargs):
        """
        Build the architecture of the cascade
        input:
            more_neg_path: path to directory of negative samples
            kwargs: arguments for FaceDetectorFeatures
        output:
            cascaded_classifier: the cascade classifier
        """
        ## F0 = 1.0 
        F1 = 1.0 ##
        ## D0 = 1.0
        D1 = 1.0 ##
        i = 0 # layer number        
        cascaded_classifier = CascadeClassifier(self.X_train, self.y_train, 
                                                verbose=self.verbose, layers=[], 
                                                batchsize=self.batchsize, use_stored=False)
        
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
            strong_classifier_chooser = StrongClassifierChooser(self.X_train, self.y_train, n_i, verbose=self.verbose , 
                                                                batchsize=self.batchsize, delete_unused=self.delete_unused, 
                                                                equal_weights=self.equal_weights)
            while F1 > self.f * F0:
                n_i += 1

                if self.verbose:
                    print(f"Building layer {i}, classifier {n_i}")
                strong_classifier_chooser.T = n_i
                # observe that adding strong classifiers repeats job except for the last one, so we can save time by not repeating and starting from the last one
                strong_classifier = strong_classifier_chooser.train(n_i - 1)
                # TODO check if use cascade_classifier or strong_classifier
                confidence = strong_classifier.confidence(self.X_val)
                cur_preds = confidence >= strong_classifier.θ
                D1_temp = np.sum((cur_preds == 1) & (self.y_val == 1)) / np.sum(self.y_val == 1)
                if D1_temp < D0 * self.d:
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
                # TODO use cascaded_classifier or strong_classifier or it doesn't matter?
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
                size_before = self.X_train.shape[1]
                self.X_train = self.X_train[:, remaining_bool_train]
                self.y_train = self.y_train[remaining_bool_train]
                self.X_val = self.X_val[:, remaining_bool_val]
                self.y_val = self.y_val[remaining_bool_val]

                if more_neg_path is not None:
                    zeros_cnt = np.sum(self.y_train == 0)
                    req_cnt = kwargs.get('req_cnt', 6000)
                    req_cnt = req_cnt - zeros_cnt
                    kwargs['req_cnt'] = req_cnt
                    kwargs['classifier'] = cascaded_classifier

                    ret_cnt = self.getMoreNeg(more_neg_path, *args, **kwargs)
                    print(f"Added {ret_cnt} negative samples")

            if self.verbose:
                print(f"Layer {i} built, false positive rate: {F0}, detection rate: {D0}, number of classifiers: {n_i}")
            
            if i >= self.maxlayers:
                print("Max number of layers reached, F1: ", F1)
                break


        if self.verbose:
            print("Architecture built")

        for strong_classifier in self.strong_classifiers:
            cascaded_classifier.layers.append(len(strong_classifier.weak_classifiers))
            
        return cascaded_classifier

    def getMoreNeg(self, 
                   more_neg_path: str, 
                   n_per_img: int = 6,
                   resize_factor: float = 1,
                   req_cnt: int = 6000,
                   by_confidence: bool = False,
                   by_size: bool = False,
                   *args, **kwargs):
        """
        Get more negative samples from more_neg_path
        negative samples are from non-faces directory that are classified as faces (false positives)
        input:
            more_neg_path: path to directory of negative samples
            n_per_img: number of negative samples to get from each image   
            req_cnt: required number of negative samples
            by_confidence: if True, get negative samples by confidence
            by_size: if True, get negative samples by size
            kwargs: arguments for FaceDetectorFeatures
        output:
            chosen_features: chosen negative samples
        """
        from detect_face import FaceDetectorFeatures
        face_detector = FaceDetectorFeatures(*args, **kwargs)
        # shuffle files
        files = glob.glob(more_neg_path + "/*.png")
        np.random.shuffle(files)
        cnt = 0
        # chosen_features = np.zeros((self.n_features, 0))
        req_cnt_per_img = n_per_img
        for i in tqdm(range(len(files)), desc='Images', colour='yellow'):
            img = cv2.imread(files[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (int(img.shape[1] * resize_factor), int(img.shape[0] * resize_factor)))
            features = face_detector.find_face_features(img, n_faces=req_cnt_per_img, by_confidence=by_confidence, by_size=by_size)  # (n_features, n_faces)
            if features.shape[1] == 0:
                continue
            if features.shape[1] > 1:

                # split to train and val
                features_train, features_val = train_test_split(features.T, test_size=self.v_size, random_state=42)
                features_train = features_train.T
                features_val = features_val.T
                self.X_val = np.concatenate((self.X_val, features_val), axis=1)
                self.y_val = np.concatenate((self.y_val, np.zeros(features_val.shape[1], dtype=int)), axis=0)
            else:
                features_train = features

            # add to X_train, y_train, X_val, y_val
            self.X_train = np.concatenate((self.X_train, features_train), axis=1)
            self.y_train = np.concatenate((self.y_train, np.zeros(features_train.shape[1], dtype=int)), axis=0)
            cnt += features.shape[1]
            if cnt >= req_cnt: 
                break

            if features.shape[1] < req_cnt_per_img:
                req_cnt_per_img += req_cnt_per_img - features.shape[1]
            else:
                req_cnt_per_img = n_per_img

        return cnt
            
        

    
    