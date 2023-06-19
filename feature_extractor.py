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
    device: device to run the model
    verbose: print out information

    all_features_file: file to save all features
    selected_features_file: file to save selected features
    indecies_file: file to save indecies of selected features
    labels_file: file to save labels of selected features
    """
    def __init__(self, 
                 shape, 
                 percentile=10, 
                 batch_size=500, 
                 device=None, 
                 verbose=True,
                 all_features_file="all_features.npz",
                 selected_features_file="selected_features.npz",
                 indecies_file="indecies.npy",
                 labels_file="labels.npy",):
        self.shape = shape
        self.percentile = percentile
        self.batch_size = batch_size
        self.f2, self.f3, self.f4 = self.describe_features(shape)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device
        self.verbose = verbose

        self.all_features_file = all_features_file
        self.selected_features_file = selected_features_file
        self.indecies_file = indecies_file
        self.labels_file = labels_file


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
                        transform=None,
                        save_to_file=True):
        """
        Reads dataset, extract features and return X matrix and y vector
        Used to load training data
        transform: transform to apply to images

        pos_path: path to positive images
        neg_path: path to negative images
        
        if save_to_file is True,
        saves X (features) and y (labels) to files

        return X, y (all features and labels)
        """
        dataset = self.readDataset(pos_path, neg_path, transform=transform)
        dataset_ii = self.getIntegralImage(dataset[0]), dataset[1]
        all_features = self.getFeaturesFromDesc(self.f2, self.f3, self.f4, dataset_ii)
        if self.verbose:
            print('all_features shape:', all_features.shape)
            if save_to_file:
                print('Now, saving all features to', self.all_features_file)
        
        if save_to_file:
            np.savez_compressed(self.all_features_file, all_features)
            np.save(self.labels_file, dataset[1])

        return all_features, dataset[1]

    def extractFeatures2(self, 
                         imgs: torch.Tensor,
                         create_ii=False,
                         use_percentile=True,):
        """
        different from extractFeatures in that it does not read dataset from files, it takes images
            also, it does not take a classifier to choose features, it uses all features
        imgs: list of images
        create_ii: if True, create integral images from imgs
        use_percentile: if True, use selectPercentile to select features
        return: features of imgs (n_features, n_images)
        """
        if create_ii:
            imgs = self.getIntegralImage(imgs)

        if use_percentile:
            indecies, _ = self.loadPercentileIndecies()
            f2, f3, f4 = self._idx2f_desc(self.f2, self.f3, self.f4, indecies)
        else:
            f2, f3, f4 = self.f2, self.f3, self.f4

        all_features = self.getFeaturesFromDesc(f2, f3, f4, (imgs, None))

        return all_features

    def readDataset(self, pos_path, neg_path, transform=None):
        """
          return np array of (img, label)
          transform: transform to apply to images
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
            if transform is not None:
                img = transform(img)
            img = cv2.resize(img, self.shape)
            imgs.append(img)
            labels.append(1)
        for i in range(len(neg_img_names)):
            img = cv2.imread(neg_img_names[i], 0)
            if transform is not None:
                img = transform(img)
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

    def extractFeaturesByClassifier(self,
                                    pos_path,
                                    neg_path,
                                    cascadeClassifier: CascadeClassifier,
                                    transform=None):
        """
        Same as extractFeaturesByIndecies to get indecies from cascadeClassifier - just a better name
        """
        return self.extractFeaturesByIndecies(pos_path, neg_path, cascadeClassifier, transform)
    
    def extractFeaturesByIndecies(self,
                                  pos_path,
                                  neg_path,
                                  cascadeClassifier: CascadeClassifier,
                                  transform=None):
        """
        Used to extract features from dataset (example: testset) by indecies of features chosen by cascadeClassifier
        
        if cascadeClassifier is not None, get indecies from all WeakClassifiers in cascadeClassifier
        
        It needs selectPercentile to be run before because selectPercentile selects features with highest F-value, then cascadeClassifier chooses features from them
        
        transform: transform to apply to images
        return 
            indecies: (dict): indecies[f_idx] = indecies of f_idx-th feature in all_features.
                      values of f_idx are chosen by cascadeClassifier
            all_features: features of dataset (n_features, n_images)
            labels: labels of dataset (n_images)
        """

        m_indecies = self.getChosenIndeceies(cascadeClassifier)

        if m_indecies.shape[0] == 0:
            raise ValueError('indecies is empty')

        if self.verbose:
            print('Now reading dataset...')
        dataset = self.readDataset(pos_path, neg_path, transform=transform)
        dataset_ii = self.getIntegralImage(dataset[0]), dataset[1]
        
        if self.verbose:
            print('Now extracting features from dataset...')
        # indecies of selectPercentile features
        # p_indecies, _ = self.selectPercentile()
        # TODO memoize the following
        p_indecies = self.loadPercentileIndecies()

        f2_p, f3_p, f4_p = self._idx2f_desc(self.f2, self.f3, self.f4, p_indecies)
        f2_m, f3_m, f4_m = self._idx2f_desc(f2_p, f3_p, f4_p, m_indecies)

        all_features = self.getFeaturesFromDesc(f2_m, f3_m, f4_m, dataset_ii)
        f_locations = dict()
        for i in range(all_features.shape[0]):
            f_locations[m_indecies[i]] = i

            
        return f_locations, all_features, dataset_ii[1]
        

    def _idx2f_desc(self, f2, f3, f4, indecies):
        """
        return features descriptions by indecies
        """
        f2_desc = f2[indecies[indecies < f2.shape[0]]]
        f3_desc = f3[indecies[(indecies >= f2.shape[0]) & (indecies < f2.shape[0] + f3.shape[0])] - f2.shape[0]]
        f4_desc = f4[indecies[indecies >= f2.shape[0] + f3.shape[0]] - f2.shape[0] - f3.shape[0]]
        return f2_desc, f3_desc, f4_desc
    

    def getChosenIndeceies(self, cascadeClassifier: CascadeClassifier):
        """
        return indecies of features chosen by cascadeClassifier
        """
        m_indecies = []
        for strong_classifier in cascadeClassifier.strong_classifiers:
            for weak_classifier in strong_classifier.weak_classifiers:
                m_indecies.append(weak_classifier.feature_index)
        m_indecies = np.array(m_indecies)
        m_indecies = np.unique(m_indecies)
        return m_indecies
    

    def extractFeaturesFromImage(self, img: torch.Tensor,
                                 cascadeClassifier: CascadeClassifier,
                                 transform=None):
        """
        Used to extract features from image by indecies of features chosen by cascadeClassifier

        if cascadeClassifier is not None, get indecies from all WeakClassifiers in cascadeClassifier
        """
        m_indecies = self.getChosenIndeceies(cascadeClassifier)

        if m_indecies.shape[0] == 0:
            raise ValueError('indecies is empty')
        
        if self.verbose:
            print('Now extracting features from image...')
        # indecies of selectPercentile features
        # p_indecies, _ = self.selectPercentile()
        p_indecies = self.loadPercentileIndecies()

        f2_p, f3_p, f4_p = self._idx2f_desc(self.f2, self.f3, self.f4, p_indecies)
        f2_m, f3_m, f4_m = self._idx2f_desc(f2_p, f3_p, f4_p, m_indecies)
        
        if transform is not None:
            img = transform(img)
        
        ii = self.getIntegralImage(img).numpy() # converted later to torch.Tensor

        all_features = self.getFeaturesFromDesc(f2_m, f3_m, f4_m, (ii, None))
        f_locations = dict()
        for i in range(all_features.shape[0]):
            f_locations[m_indecies[i]] = i

        return f_locations, all_features

    
    def selectPercentile(self, X=None, y=None, saveMemory=True, save_to_file=True):
        """
        Selects features with highest F-value
        X: features (n_features, n_images)
        y: labels (n_images)
        if X and y are None, load them from files

        percentile: percentile of features to select
        return indecies of selected features, selected_featues

        if file 'indecies.npy' exists, load indecies from file
        else, calculate indecies and save them to file
        """
        if X is None or y is None:
            if os.path.exists(self.indecies_file) and os.path.exists(self.selected_features_file):
                indecies = np.load(self.indecies_file)
                X = np.load(self.selected_features_file)['arr_0']
                return indecies, X
            
            if saveMemory:
                del self.f2
                del self.f3
                del self.f4
                
            if not os.path.exists(self.all_features_file) or not os.path.exists(self.labels_file):
                raise ValueError('all_features_file or labels_file does not exist please run extractFeatures first')
            

            X = np.load(self.all_features_file)['arr_0']
            y = np.load(self.labels_file)


        if self.verbose:
            print('Now selecting percentile features')

        selector = SelectPercentile(f_classif, percentile=self.percentile)
        indecies = selector.fit(X.T, y).get_support(indices=True)

        if save_to_file:
            np.save(self.indecies_file, indecies)
            np.savez(self.selected_features_file, X[indecies])

        if saveMemory:
            self.f2, self.f3, self.f4 = self.describe_features(self.shape)
        return indecies, X[indecies]
    
    def loadPercentileIndecies(self):
        """
        load indecies of selected features from file
        """
        if os.path.exists(self.indecies_file):
            return np.load(self.indecies_file)
        else:
            raise ValueError('indecies_file does not exist, please run selectPercentile first')