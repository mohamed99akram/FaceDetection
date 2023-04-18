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
    def __init__(self, shape, percentile=10, batch_size=500, device=None, verbose=True):
        self.shape = shape
        self.percentile = percentile
        self.batch_size = batch_size
        self.f2, self.f3, self.f4 = self.describe_features(shape)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.verbose = verbose

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

                        saveAllto='all_features.npz',
                        saveYto='labels.npy',
                        saveIndeciesto='indecies.npy',
                        saveSelectedto='all_features_10.npz'):
        """
        Reads dataset, extract features and return X matrix and y vector
        Used to load training data

        pos_path: path to positive images
        neg_path: path to negative images
        
        saveAllto: path to save features
        saveYto: path to save labels
        saveIndeciesto: path to save indecies of selected features
        saveSelectedto: path to save selected features

        return 
        X matrix (n_features, n_images), 
        y vector (n_images)
        indecies of selected features
        selected percentile of features
        """
        dataset = self.get_dataset(pos_path, neg_path)
        dataset_ii = self.getIntegralImage(dataset[0]), dataset[1]
        all_features = self.getFeaturesFromDesc(self.f2, self.f3, self.f4, dataset_ii)
        if self.verbose:
            print('all_features shape:', all_features.shape)
            print('Now, saving all features to', saveAllto)

        np.savez_compressed(saveAllto, all_features)
        np.save(saveYto, dataset[1])

        if self.verbose:
            print('Now selecting percentile features')
        
        indecies = None
        if saveSelectedto is not None:
            indecies, selectedPercentile = self.selectPercentile(X=all_features, 
                                             y=dataset[1], 
                                             percentile=self.percentile,
                                             saveIndeciesTo=saveIndeciesto, 
                                             saveFeaturesTo=saveSelectedto)

        return all_features, dataset[1], indecies, selectedPercentile


    def get_dataset(self, pos_path, neg_path):
        """
          return np array of (img, label)
        """
        pos_img_names = os.listdir(pos_path)
        neg_img_names = os.listdir(neg_path)
        for i in range(len(pos_img_names)):
            pos_img_names[i] = os.path.join(pos_path, pos_img_names[i])
        for i in range(len(neg_img_names)):
            neg_img_names[i] = os.path.join(neg_path, neg_img_names[i])

        pos_img_names = np.array(pos_img_names)
        neg_img_names = np.array(neg_img_names)
    
        imgs, labels = [], []
        for i in range(len(pos_img_names)):
            img = cv2.imread(pos_img_names[i], 0)
            img = cv2.resize(img, self.shape)
            imgs.append(img)
            labels.append(1)
        for i in range(len(neg_img_names)):
            img = cv2.imread(neg_img_names[i], 0)
            img = cv2.resize(img, self.shape)
            imgs.append(img)
            # labels.append(-1)
            labels.append(0)
        return np.array(imgs), np.array(labels)


    def getFeaturesFromDesc(self, f2, f3, f4, dataset_ii):
        """
        f2: descriptions of 2-rect features
        f3: descriptions of 3-rect features
        f4: descriptions of 4-rect features
        dataset_ii: integral images of dataset
        """
        class FeatureDataset(Dataset):
            """
            Dataset of features

            """
            def __init__(self, f):
                if type(f) == list or type(f) == np.ndarray:
                    self.f = torch.from_numpy(np.array(f)).type(torch.int64)
                else:
                    self.f = f.type(torch.int64)

            def __len__(self):
                return self.f.shape[0]
            def __getitem__(self, idx):
                return self.f[idx]
            
        
        f2d = FeatureDataset(f2)
        f3d = FeatureDataset(f3)
        f4d = FeatureDataset(f4)

        f2d_loader = DataLoader(f2d, batch_size=self.batch_size)
        f3d_loader = DataLoader(f3d, batch_size=self.batch_size)
        f4d_loader = DataLoader(f4d, batch_size=self.batch_size)

        iis = dataset_ii[0].astype(int)
        iis = torch.from_numpy(iis).to(self.device)

        ii_f2 = torch.zeros((iis.shape[0], f2.shape[0]))
        ii_f3 = torch.zeros((iis.shape[0], f3.shape[0]))
        ii_f4 = torch.zeros((iis.shape[0], f4.shape[0]))

        for i, f2_b in enumerate(f2d_loader):
            f2_b = f2_b.to(self.device)
            ii_f2[:, i*self.batch_size:(i+1)*self.batch_size] = (self.getFeatureValue(iis, f2_b[:, 0]) - self.getFeatureValue(iis, f2_b[:,1])).to('cpu')

        for i, f3_b in enumerate(f3d_loader):
            f3_b = f3_b.to(self.device)
            ii_f3[:, i*self.batch_size:(i+1)*self.batch_size] = (-self.getFeatureValue(iis, f3_b[:, 0]) + self.getFeatureValue(iis, f3_b[:,1]) - self.getFeatureValue(iis, f3_b[:,2])).to('cpu')

        for i, f4_b in enumerate(f4d_loader):
            f4_b = f4_b.to(self.device)
            ii_f4[:, i*self.batch_size:(i+1)*self.batch_size] = (-self.getFeatureValue(iis, f4_b[:, 0]) + self.getFeatureValue(iis, f4_b[:,1]) + self.getFeatureValue(iis, f4_b[:,2]) - self.getFeatureValue(iis, f4_b[:,3])).to('cpu')

        all_features = torch.cat((ii_f2, ii_f3, ii_f4), dim=1)
        all_features = all_features.t()
        all_features = all_features.numpy()
        return all_features


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
                         X_file='all_features.npz', 
                         y_file='labels.npy', 
                         saveFeaturesTo='all_features_10.npz', 
                         saveIndeciesTo='indecies.npy'):
        """
        Selects features with highest F-value

        percentile: percentile of features to select
        X_file: path to features
        y_file: path to labels
        saveFeaturesTo: path to save selected features
        saveIndeciesTo: path to save indecies of selected features

        return indecies of selected features, selected_featues

        if file 'indecies.npy' exists, load indecies from file
        else, calculate indecies and save them to file
        """
        if os.path.exists(saveIndeciesTo) and os.path.exists(saveFeaturesTo):
            indecies = np.load(saveIndeciesTo)
            X = np.load(saveFeaturesTo)['arr_0']
            return indecies, X
        
        
        if X is None or y is None:
            if not os.path.exists(X_file) or not os.path.exists(y_file):
                raise ValueError('X_file or y_file does not exist')
            X = np.load(X_file)['arr_0']
            y = np.load(y_file)


        selector = SelectPercentile(f_classif, percentile=percentile)
        indecies = selector.fit(X.T, y).get_support(indices=True)

        np.save(saveIndeciesTo, indecies)
        np.savez(saveFeaturesTo, X[indecies])
        return indecies, X[indecies]