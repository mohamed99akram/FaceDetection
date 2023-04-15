# >>>>>>>
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class OneClassifier:
    def __init__(self, feature_index, feature_val, threshold, polarity, error):
        self.feature_index = feature_index
        self.feature_val = feature_val
        self.threshold = threshold
        self.polarity = polarity
        self.error = error

    # make a function for easier access as numpy array, example: np.array(wc)
    def __array__(self):
        # return tensor.cpu() if members are tensors else np.array
        if type(self.feature_index) == torch.Tensor:
            return np.array([self.feature_index.cpu().numpy(), self.feature_val.cpu().numpy(), self.threshold.cpu().numpy(), self.polarity.cpu().numpy(), self.error.cpu().numpy()])
        else:
            return np.array([self.feature_index, self.feature_val, self.threshold, self.polarity, self.error])

    def __str__(self):
        return np.array(self).__str__()

class WeakClassifier():
    """
    Weak classifier
    Chooses a feature and a threshold to classify the data
    X: features matrix of shape (n_features, n_samples)
    y: labels vector of shape (n_samples,): either 0 or 1
    weights: weights vector of shape (n_samples,): they are positive and sum to 1
    batchsize: batchsize for dataloader
    show_time: show time for each step
    show_mem: show memory for each step
    delta: to take threshold below feature value -> θ = f - δ

    TODO: consider taking (f_i+1 + f_i) / 2 as threshold
    """
    def __init__(self,
                  X:torch.Tensor or np.ndarray,
                  y:torch.Tensor or np.ndarray,
                  weights: torch.Tensor or np.ndarray,
                  batchsize, 
                  show_time=False, 
                  show_mem=False,
                  delta=0.00001):

        #+ Device
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.show_time = show_time
        self.show_mem = show_mem
        self.delta = delta

        # self.X = X
        #+ Dataset
        # make sure y is 0 or 1
        assert (y == 0).sum() + (y == 1).sum() == y.shape[0], 'y should be 0 or 1'
        if type(y) == torch.Tensor:
            self.y = y.type(torch.float32).to(self.device)
        else:
            self.y = torch.tensor(y).type(torch.float32).to(self.device)
        # X
        if type(X) == torch.Tensor:
            self.X = X.type(torch.float32)
        else:
            self.X = torch.tensor(X).type(torch.float32)

        self.dataset = WeakClassifier._FeaturesDataset(self.X)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=batchsize, num_workers=2)
        self.batchsize = batchsize
        if type(weights) == torch.Tensor:
            self.weights = weights.type(torch.float32).to(self.device)
        else:
            self.weights = torch.tensor(weights).type(torch.float32).to(self.device)
            

    def chooseClassifier(self):
        """
        Chooses a classifier
        Returns:
            best_index: index of the feature
            best_threshold: threshold of the feature
            best_polarity: polarity of the feature
            best_error: error of the feature
        """
        
        s_t = time.time()

        classifiers4 = []

        best_best_index = 0
        best_best_threshold = 0
        best_best_polarity = 1
        best_best_error = float('inf')

        overall_index = 0
        LW, RW = [], []
        for index, X in enumerate(self.dataloader):

            if self.show_time:
                print('At batch number: ', index, ':', ' Start time: ', time.time() - s_t)
    
            X = X.to(self.device)

            if self.show_mem:
                print('Memory for batch: ', self._mem())
            

            min_error = torch.tensor([float('inf')]*X.shape[0],device=self.device)
            best_feature = torch.zeros((X.shape[0], 2), device=self.device)
            best_threshold = torch.zeros(X.shape[0], device=self.device)
            best_polarity = torch.zeros(X.shape[0], device=self.device)

            #+ repeats the weights and labels for each feature (each row of X)
            weights2d = torch.tile(self.weights, (X.shape[0], 1))
            # del weights
            y2d = torch.tile(self.y, (X.shape[0], 1))
            # del y

            #+ sort features, labels, weights by corresponding features (sort each row of X)
            sorting_indecies = torch.argsort(X, stable=True)
            idx0 = torch.arange(X.shape[0]).reshape(-1, 1).to(self.device) #% (n_features_sub, 1)

            #+ s_w: srorted weights, s_f: sorted features, s_y: sorted labels
            s_w = weights2d[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            # del weights2d
            s_f = X[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            # del X
            s_y = y2d[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            # del y2d
            # del idx0

            if self.show_mem:
                self._mem()

            # for left, right in [(-1, 1), (1, -1)]:  # @ y: -1 or 1
            for left, right in [(0, 1), (1, 0)]:  # @ y: 0 or 1

                #+ left_weights, right_weights: accumulative weights for each feature (each row of X) left and right
                left_weights = self._accW_left(s_w, s_y, left)
                right_weights = self._accW_right(s_w, s_y, right)

                LW.append(left_weights)
                RW.append(right_weights)

                #+ idx: index of the feature that has the minimum error
                idx = torch.argmin(left_weights + right_weights, axis=1) #% (n_features_sub,)
                # seems problamtic
                idx[idx >= s_f.shape[1]] = s_f.shape[1] - 1
                ii1 = torch.arange(idx.shape[0], device=self.device) #% same as idx0? (n_features_sub,)

                #+ min_error = min(left_weights + right_weights, min_error)
                cur_min_error = left_weights[ii1, idx] + right_weights[ii1, idx] #% (n_features_sub,)
                temp_bool = cur_min_error < min_error #% (n_features_sub,)
                if temp_bool.any().item():
                    min_error[temp_bool] = cur_min_error[temp_bool] #% (n_features_sub,)

                    selected_idx = idx[temp_bool] #% (only better indeces,)

                    selected_features = s_f[ii1[temp_bool], selected_idx] #% (n_features_sub,) => update only better indecies

                    #???? USELESS LINE ?????
                    # best_feature[temp_bool] = torch.tensor(
                    #     list(zip(selected_idx, selected_features)), device=self.device) #% (n_features_sub, 2) => update only better indecies
                    best_feature[temp_bool] = torch.cat((selected_idx.reshape(-1, 1), selected_features.reshape(-1, 1)), axis=1)

                    best_threshold[temp_bool] = s_f[ii1[temp_bool], idx[temp_bool]] - self.delta  #% (n_features_sub,) => update only better indecies
                    # best_polarity[temp_bool] = left  #% (n_features_sub,) => update only better indecies
                    best_polarity[temp_bool] = -1 if left == 0 else 1  #% (n_features_sub,) => update only better indecies
                    # best_polarity[temp_bool] = 1 if left == 0 else -1  #% (n_features_sub,) => update only better indecies
                
            # + add to classifiers4, converted to numpy
            classifiers4.extend([OneClassifier(*clf4) for clf4 in zip(best_feature[:, 0],
                                best_feature[:, 1], best_threshold, best_polarity, min_error)]) 
            
            #+ select overall best classifier
            current_best_best_error_index = torch.argmin(min_error)
            if min_error[current_best_best_error_index] < best_best_error:
                best_best_index = overall_index + current_best_best_error_index
                best_best_threshold = best_threshold[current_best_best_error_index]
                best_best_polarity = best_polarity[current_best_best_error_index]
                best_best_error = min_error[current_best_best_error_index]

            overall_index += X.shape[0]


        if self.show_time:
            print("Time taken: %f seconds" % (time.time() - s_t))

        return best_best_index, best_best_threshold, best_best_polarity, best_best_error, classifiers4, LW, RW

    def _accW_left(self, s_w, s_y, left):
        zero_col = torch.zeros((s_w.shape[0], 1), device=self.device) #% (n_features_sub, 1)
        # chosenW = s_w * (s_y == left) #% (n_features_sub, n_samples)
        chosenW = s_w * (s_y != left) #% (n_features_sub, n_samples)
        accW = torch.cumsum(chosenW, axis=1) #% (n_features_sub, n_samples) 
        accW = torch.cat((zero_col, accW), axis=1) #% (n_features_sub, n_samples + 1)
        return accW
    

    def _accW_right(self, s_w, s_y, right):
        zero_col = torch.zeros((s_w.shape[0], 1), device=self.device) #% (n_features_sub, 1)
        chosenW = s_w * (s_y != right) #% (n_features_sub, n_samples)
        # chosenW = s_w * (s_y == right) #% (n_features_sub, n_samples)
        rev_W = torch.flip(chosenW, dims=[1]) #% (n_features_sub, n_samples)
        accW = torch.cumsum(rev_W, axis=1) #% (n_features_sub, n_samples)
        accW = torch.flip(accW, dims=[1])   #% (n_features_sub, n_samples)
        accW = torch.cat((accW, zero_col), axis=1) #% (n_features_sub, n_samples + 1)
        return accW

    def _mem(self):
        """
        Returns allocated and reserved memory in Mb
        """
        return torch.cuda.memory_allocated()/(1024**2), torch.cuda.memory_reserved()/(1024**2)
              

    class _FeaturesDataset(Dataset):
        """
        Dataset for features
        """
        def __init__(self, X):
            self.X = X

        def __getitem__(self, index):
            return self.X[index]

        def __len__(self):
            return len(self.X)
