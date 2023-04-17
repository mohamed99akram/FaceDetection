# feature description
# select percentile
# feature extraction (trainig)
# feature extraction (testing)
# feature extraction (image)
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import os
from sklearn.feature_selection import SelectPercentile, f_classif
from cascade import CascadeClassifier
from classifier import WeakClassifier
from strong_classifier import StrongClassifierChooser, StrongClassifier
import pickle as pkl


class Rect():
    def __init__(self, xs, ys, xe, ye):
        self.xs = xs # x start
        self.ys = ys # y start
        self.xe = xe # x end
        self.ye = ye # y end

    def __array__(self):
        return np.array([self.xs, self.ys, self.xe, self.ye])

class FeatureExtractor:
    """
    Extract features from images
    shape: (height, width)
    percentile: percentile of features to be selected
    batch_size: batch size for feature extraction
    """
    def __init__(self, shape, percentile=None, batch_size=500):
        self.shape = shape
        self.percentile = percentile
        self.batch_size = batch_size
        self.f2, self.f3, self.f4 = self.describe_features(shape)
        

    def describe_features(self, shape):
        """
        shape: (height, width)  
        return: 2, 3, 4 features\n  
        f2: shape: (n, 2, 4), (positive, negative)  \n
        f3: shape: (n, 3, 4), (negative, positive, negative)  \n
        f4: shape: (n, 4, 4), (negative, positive, positive, negative)  
        """
        rect2 = []
        rect3 = []
        rect4 = []
        height, width = shape
        cnt = 0
        for i in range(height+1):
            for j in range(width+1):
                for k in range(1, height+1):
                    for l in range(1, width+1):
                        cnt += 1
                        # @ 2 features
                        ij = np.array([i,j,i,j])
                        l1 = np.array([(0,l,k,2*l), (0,0,k,l)])
                        l2 = np.array([(0,0,k,l), (k,0,2*k,l)])
                        # Horizontal [-][+]
                        if i + k <= height and j + 2 * l <= width:
                            rect2.append(((ij+l1[0]), (ij+l1[1]))) # p, n

                        # Vertical #+
                            # -
                        if i + 2 * k <= height and j + l <= width:
                            rect2.append(((ij+l2[0]), (ij+l2[1]))) # p, n

                        # @ 3 features
                        l3 = np.array([(0,0,k,l), (0,l,k,2*l), (0,2*l,k,3*l)])
                        l4 = np.array([(0,0,k,l), (k,0,2*k,l), (2*k,0,3*k,l)])
                        # Horizontal [-][+][-]
                        if i + k <= height and j + 3 * l <= width:
                            rect3.append(((ij+l3[0]), (ij+l3[1]), (ij+l3[2]))) # n, p, n

                        # Vertical #-
                            # +
                            # -
                        if i + 3 * k <= height and j + l <= width:
                            rect3.append(((ij+l4[0]), (ij+l4[1]), (ij+l4[2])))# n, p, n

                        # @ 4 features
                        l5 = np.array([(0,0,k,l), (0,l,k,2*l),(k, 0, 2*k, l), (k, l, 2*k, 2*l)]) # n, p, p, n
                        # [-][+]
                        # [+][-]
                        if i + 2 * k <= height and j + 2 * l <= width:
                            rect4.append(((ij+l5[0]), (ij+l5[1]), (ij+l5[2]), (ij+l5[3])))
        return np.array(rect2), np.array(rect3), np.array(rect4)


    def getIntegralImage(self, img: torch.Tensor or np.ndarray):
        if isinstance(img, torch.Tensor):
            if len(img.shape) == 2:
                ret = torch.zeros(img.shape[0]+1, img.shape[1]+1)
                ret[1:, 1:] = img
                return ret.cumsum(dim=0).cumsum(dim=1)
            ret = torch.zeros(img.shape[0], img.shape[1]+1, img.shape[2]+1)
            ret[:, 1:, 1:] = img
            return ret.cumsum(dim=1).cumsum(dim=2)
        elif isinstance(img, np.ndarray):
            if len(img.shape) == 2:
                ret = np.zeros((img.shape[0]+1, img.shape[1]+1))
                ret[1:, 1:] = img
                return np.cumsum(np.cumsum(ret, axis=0), axis=1)
            ret = np.zeros((img.shape[0], img.shape[1]+1, img.shape[2]+1))
            ret[:, 1:, 1:] = img
            return np.cumsum(np.cumsum(ret, axis=1), axis=2)


    def getFeatureValue(self, ii: np.ndarray or torch.Tensor, f: np.ndarray):
        """
        ii: integral image or list of integral images (shape: (β+1, β+1) or (l, β+1, β+1), l=number of images), β=width or height of image \n
        
        f: tensor of shape (n, 4), n=number of features (in batch), 4=number of coordinates of feature \n
        f[i] = [i1, j1, i2, j2] \n
        i1, j1, i2, j2: coordinates of feature \n
        i1, j1: top-left corner \n
        i2, j2: bottom-right corner \n
        1: i1, j1
        2: i1, j2
        3: i2, j1
        4: i2, j2
        ans = 4 + 1 - 2 - 3
        """
        i1,j1,i2,j2 = f[:,0], f[:,1], f[:,2], f[:,3]
        
        if isinstance(ii, torch.Tensor):
            if len(ii.shape) == 2:
                return ii[i2, j2] + ii[i1, j1] - ii[i1, j2] - ii[i2, j1]
            return ii[:, i2, j2] + ii[:, i1, j1] - ii[:, i1, j2] - ii[:, i2, j1] # shape: (l, 19+1, 19+1), l=number of images
        elif isinstance(ii, np.ndarray):
            if len(ii.shape) == 2:
                return ii[i2, j2] + ii[i1, j1] - ii[i1, j2] - ii[i2, j1]
            return ii[:, i2, j2] + ii[:, i1, j1] - ii[:, i1, j2] - ii[:, i2, j1]
        
    def extractFeatures(self, 
                        pos_path, 
                        neg_path, 

                        indecies=None, 
                        cascadeClassifier: CascadeClassifier=None,

                        saveAllto='all_features.npz',
                        saveYto='labels.npy',
                        saveIndeciesto='indecies.npy',
                        saveSelectedto='all_features_10.npz'):
        """
        Reads dataset, extract features and return X matrix and y vector
        Used to load training data

        pos_path: path to positive images
        neg_path: path to negative images
        indecies: indecies of features to extract (coming from trained model after selectPercentile())
        saveAllto: path to save features
        saveYto: path to save labels
        saveIndeciesto: path to save indecies of selected features
        saveSelectedto: path to save selected features

        return X matrix (n_features, n_images), y vector (n_images)
        """
        pass

    def extractFeaturesByIndecies(self,
                                  pos_path,
                                  neg_path,
                                  indecies=None,
                                  cascadeClassifier: CascadeClassifier=None,):
        """
        return chosen featues (n_sub_features, n_images), y vector (n_images), indecies of features
        Used for testing 
        
        if cascadeClassifier is not None, get indecies from all WeakClassifiers in cascadeClassifier
        else, get indecies from indecies parameter
        """
        pass



    def extractFeaturesFromImage(self, img: np.ndarray or torch.Tensor):
        pass

    
    def selectPercentile(self, 
                         percentile=10, 
                         X=None,
                         y=None,
                         X_path='all_features.npz', 
                         y_path='labels.npy', 
                         saveFeaturesTo='all_features_10.npz', 
                         saveIndeciesTo='indecies.npy'):
        """
        Selects features with highest F-value

        percentile: percentile of features to select
        X_path: path to features
        y_path: path to labels
        saveFeaturesTo: path to save selected features
        saveIndeciesTo: path to save indecies of selected features

        return indecies of selected features

        if file 'indecies.npy' exists, load indecies from file
        else, calculate indecies and save them to file
        """
        if os.path.exists('indecies.npy'):
            return np.load('indecies.npy')
        
        if X is None or y is None:
            X = np.load(X_path)['arr_0']
            y = np.load(y_path)


        selector = SelectPercentile(f_classif, percentile=percentile)
        indecies = selector.fit(X, y).get_support(indices=True)

        np.save(saveIndeciesTo, indecies)
        np.savez(saveFeaturesTo, X[indecies])
        return indecies