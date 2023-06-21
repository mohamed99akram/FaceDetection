# >>>>>>>
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict

def EQ(x, y, permittivity=1e-6):
    return np.abs(x - y) < permittivity
def EQ3(x, y, z, permittivity=1e-4):
    return EQ(x, y, permittivity) and EQ(y, z, permittivity)
def OK(msg='OK'):
    print("\033[32m{}\033[0m".format(msg))
def NOK(msg='Not Equal'):
    print("\033[31m{}\033[0m".format(msg))

class WeakClassifier():
    """
    Weak classifier
    feature_index: index of feature
    threshold: threshold of feature
    polarity: polarity of feature
    error: error of feature
    """
    def __init__(self, feature_index, threshold, polarity, error):
        # convert to cpu if tensors, convert to numbers if tensors
        self.f_idx = feature_index.cpu().numpy() if type(feature_index) == torch.Tensor else feature_index
        self.θ = threshold.cpu().numpy() if type(threshold) == torch.Tensor else threshold
        self.p = polarity.cpu().numpy() if type(polarity) == torch.Tensor else polarity
        self.ϵ = error.cpu().numpy() if type(error) == torch.Tensor else error
        self.updatedIndecies = False
        self.P = 1
        self.N = 0
        
        
        
    @property
    def feature_index(self): return self.f_idx
    @property
    def threshold(self): return self.θ
    @property
    def polarity(self): return self.p
    @property
    def error(self): return self.ϵ
    
    def predict(self, X, f_idx_map:Dict[int, int] = None):
        """
        Predicts the class of the given data X

        :param X: data to predict (n_features, n_samples) or (n_samples,) if f_given
        :param f_given: if True, X is the feature vector (n_samples,) 
                        else X is the whole data matrix (n_features, n_samples)
        
        :return: predicted class (n_samples,): 1 or 0
        """
        if self.f_idx is None or self.θ is None or self.p is None or self.ε is None:
            raise Exception("Please Call chooseClassifier() first")
        # 1 if data[best_index] * best_polarity <= best_threshold * best_polarity else 0 as numpy
        
        if f_idx_map is not None:
            return np.where(X[f_idx_map[int(self.f_idx)]] * self.p <= self.θ * self.p, 1, 0)
        return np.where(X[self.f_idx] * self.p <= self.θ * self.p, 1, 0)

    def updateIndecies(self, f_idx_map: Dict[int, int]):
        """
        Update the indecies of features in each weak classifier
        """
        if self.updatedIndecies:
            return
        self.f_idx = f_idx_map[int(self.f_idx)]
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
        return np.where(X[self.f_idx] * self.p <= self.θ * self.p, self.P, self.N)
    
    def changePN(self, p=1, n=0):
        self.P = p
        self.N = n

    # make a function for easier access as numpy array, example: np.array(wc)
    def __array__(self):
        # return tensor.cpu() if members are tensors else np.array
        if type(self.feature_index) == torch.Tensor:
            return np.array([self.feature_index.cpu().numpy(), self.threshold.cpu().numpy(), self.polarity.cpu().numpy(), self.error.cpu().numpy()])
        else:
            return np.array([self.feature_index, self.threshold, self.polarity, self.error])

    def __str__(self):
        return np.array(self).__str__()

class BestClassifier():
    """
    Weak classifier
    Chooses a feature and a threshold to classify the data
    
    input: 
      X: features matrix of shape (n_features, n_samples): float32
      y: labels vector of shape (n_samples,): either 0 or 1
      weights: weights vector of shape (n_samples,): they are positive and sum to 1: float32
      batchsize: batchsize for dataloader
      show_time: show time for each step
      show_mem: show memory for each step
      debug: show debug info (LW, RW)
      delete_unused: delete unused variables to save memory
      getClassifier: return classifier of each feature
      delta: to take threshold below feature value -> θ = f - δ
    
    TODO: consider taking (f_i+1 + f_i) / 2 as threshold
    TODO: X is passed multiple times, copied multiple times. how to handle this?
    """
    def __init__(self,
                  X:torch.Tensor or np.ndarray,
                  y:torch.Tensor or np.ndarray,
                  weights: torch.Tensor or np.ndarray,
                  batchsize=200, 
                  show_time=False, 
                  show_mem=False,
                  debug=False,
                  delete_unused=False,  
                  getClassifier=False,
                  delta=0.00001,
                  verbose=False):

        #+ Device
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.show_time = show_time
        self.show_mem = show_mem
        self.delta = delta
        self.debug = debug
        self.delete_unused = delete_unused
        self.getClassifier = getClassifier
        self.verbose=verbose
        self.f_idx = None # feature index
        self.θ = None # threshold
        self.p = None # polarity
        self.ϵ = None # error


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
            # if X is not float32, convert it to float32 then don't copy, put in tensor
            if X.dtype != torch.float32:
                X = X.type(torch.float32)
                self.X = X
        else:
            if type(X) == np.ndarray:
                # if X is not float32, convert it to float32 then don't copy, put in tensor
                if X.dtype != np.float32:
                    X = X.astype(np.float32)
                    self.X = torch.from_numpy(X)
                else:
                    self.X = torch.from_numpy(X)

            else:
                raise TypeError('X should be either torch.Tensor or np.ndarray')


        self.dataset = BestClassifier._FeaturesDataset(self.X)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=batchsize, num_workers=2)
        self.batchsize = batchsize
        if type(weights) == torch.Tensor:
            self.weights = weights.type(torch.float32).to(self.device)
        else:
            self.weights = torch.tensor(weights).type(torch.float32).to(self.device)
            

    def chooseClassifier(self) -> Tuple[WeakClassifier, Tuple[List[WeakClassifier], torch.Tensor, torch.Tensor]]:
        """
        Chooses a classifier
        Returns:
            best_index: index of the feature
            best_threshold: threshold of the feature
            best_polarity: polarity of the feature
            best_error: error of the feature
            classifiers4: classifiers for each feature, useful if getClassifier is True
            LW: left weights useful if debug is True
            RW: right weights useful if debug is True
        """
        
        s_t = time.time()

        classifiers4 = []

        best_best_index = 0
        best_best_threshold = 0
        best_best_polarity = 1
        best_best_error = float('inf')

        overall_index = 0
        LW, RW = [], []
        if self.verbose:
            print("Starting to choose classifier")
        for index, X in enumerate(self.dataloader):

            xshape = X.shape

            if self.show_time:
                print('At batch number: ', index, ':', ' Start time: ', time.time() - s_t)
    
            X = X.to(self.device)

            if self.show_mem:
                print('Memory for batch: ', self._mem())
            
            min_error, best_feature, best_threshold, best_polarity = float('inf'), 0, 0, 0

            #+ repeats the weights and labels for each feature (each row of X)
            weights2d = torch.tile(self.weights, (X.shape[0], 1))
            # if self.delete_unused:
            #     del self.weights
            y2d = torch.tile(self.y, (X.shape[0], 1))
            # if self.delete_unused:
            #     del self.y

            #+ sort features, labels, weights by corresponding features (sort each row of X)
            sorting_indecies = torch.argsort(X, stable=True)
            idx0 = torch.arange(X.shape[0]).reshape(-1, 1).to(self.device) #% (n_features_sub, 1)

            #+ s_w: srorted weights, s_f: sorted features, s_y: sorted labels
            s_w = weights2d[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            if self.delete_unused:
                del weights2d
            s_f = X[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            if self.delete_unused:
                del X
            s_y = y2d[idx0, sorting_indecies] #% (n_features_sub, n_samples)
            if self.delete_unused:
                del y2d
                del idx0

            if self.show_mem:
                self._mem()

            # for left, right in [(0, 1), (1, 0)]:  # @ y: 0 or 1
            left, right = 0, 1

            #+ left_weights, right_weights: accumulative weights for each feature (each row of X) left and right
            left_weights = self._accW_left(s_w, s_y, left) #% (n_features_sub, n_samples + 1)
            right_weights = self._accW_right(s_w, s_y, right) #% (n_features_sub, n_samples + 1)

            error = left_weights + right_weights #% (n_features_sub, n_samples + 1)

            #+ pol_bool: if error < 0.5: polarity = left_, else: polarity = right
            pol_bool = error < 0.5 #% (n_features_sub, n_samples + 1)

            polarity = torch.where(pol_bool, torch.tensor(left, device=self.device), torch.tensor(right, device=self.device)) #% (n_features_sub, n_samples + 1)
            polarity[polarity == 0] = -1

            #+ error = error if error < 0.5 else 1 - error: reversed polarity
            error = torch.where(pol_bool, error, 1 - error) #% (n_features_sub, n_samples + 1)
            
            #@ idx: index of the feature that has the minimum error
            idx = torch.argmin(error, axis=1) #% (n_features_sub,)
            
            #+ if out of range, set to 0: all misclassified, ∑w_l = 0, ∑w_r = ∑w is equivalent to ∑w_l = ∑w, ∑w_r = 0 with reversed polarity
            tmp_bool = idx >= s_f.shape[1]
            idx[tmp_bool] = 0
            # reverse bool
            pol_bool[tmp_bool] = ~pol_bool[tmp_bool]
            polarity[tmp_bool] = -1 if left == 0 else 1


            ii1 = torch.arange(idx.shape[0], device=self.device) #% same as idx0? (n_features_sub,)

            selected_features = s_f[ii1, idx] #% (n_features_sub,)
    
            best_threshold = s_f[ii1, idx] - self.delta #% (n_features_sub,)
            # best_threshold = (s_f[ii1, idx] + s_f[ii1, idx + 1]) / 2 #% (n_features_sub,)
            best_polarity = polarity[ii1, idx] #% (n_features_sub,)
            min_error = error[ii1, idx] #% (n_features_sub,) 


            #+ for debugging
            if self.debug:
                LW.append(left_weights)
                RW.append(right_weights)
                LW.append(self._accW_left(s_w, s_y, right))
                RW.append(self._accW_right(s_w, s_y, left))
            else:
                LW = torch.tensor([0.25, 0.25])
                RW = torch.tensor([0.25, 0.25])
            
            if self.getClassifier:
                # TODO delete selected_features from this - delete this and only use idx
                best_feature = torch.cat((idx.reshape(-1,1), selected_features.reshape(-1, 1)), axis=1) #% (n_features_sub, 2)
                # + add to classifiers4, converted to numpy
                classifiers4.extend([WeakClassifier(*clf4) for clf4 in zip(best_feature[:, 0].cpu(), best_threshold.cpu(), best_polarity.cpu(), min_error.cpu())])


            #+ select overall best classifier
            current_best_best_error_index = torch.argmin(min_error)
            if min_error[current_best_best_error_index] < best_best_error:
                best_best_index = overall_index + current_best_best_error_index
                best_best_threshold = best_threshold[current_best_best_error_index]
                best_best_polarity = best_polarity[current_best_best_error_index]
                best_best_error = min_error[current_best_best_error_index]

            overall_index += xshape[0]


        if self.show_time:
            print("Time taken: %f seconds" % (time.time() - s_t))

        # convert to cpu
        # best_best_threshold = best_best_threshold.cpu()
        # best_best_polarity = best_best_polarity.cpu()
        # best_best_error = best_best_error.cpu()
        best_best_threshold = float(best_best_threshold)
        best_best_polarity = int(best_best_polarity)
        best_best_error = float(best_best_error)
        

        self.f_idx = best_best_index
        self.θ = best_best_threshold
        self.p = best_best_polarity
        self.ε = best_best_error
        if self.delete_unused:
            del self.X
            del self.y
            del self.weights

        return WeakClassifier(best_best_index, best_best_threshold, best_best_polarity, best_best_error), (classifiers4, LW, RW)

    #########################################################################################################################

    def predict(self, X=None, f_given=False):
        """
        Predicts the class of the given data X

        :param X: data to predict (n_features, n_samples) or (n_samples,) if f_given
        :param f_given: if True, X is the feature vector (n_samples,) 
                        else X is the whole data matrix (n_features, n_samples)

        :return: predicted class (n_samples,): 1 or 0
        """
        if X is None:
            X = self.X.numpy()

        if self.f_idx is None or self.θ is None or self.p is None or self.ε is None:
            raise Exception("Please Call chooseClassifier() first")
        # 1 if data[best_index] * best_polarity <= best_threshold * best_polarity else 0 as numpy
        if f_given:
            return np.where(X * self.p <= self.θ * self.p, 1, 0)
        return np.where(X[self.f_idx] * self.p <= self.θ * self.p, 1, 0)

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
