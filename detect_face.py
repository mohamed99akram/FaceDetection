import torch
import torch.nn.functional as F
from feature_extractor import FeatureExtractor
import numpy as np
import time
from cascade import CascadeClassifier
from typing import Tuple
from sklearn.ensemble import AdaBoostClassifier

def get_subwindows(img: torch.Tensor or np.ndarray, size: Tuple, stride: int, device: torch.device):
    """
    img: torch.Tensor, shape = 
        (height, width) for a single image (GRAY),
        (n_channels, height, width) for a single image (RGB),
        (n_images, n_channels, height, width) for multiple images (RGB or GRAY)

        or np.ndarray, shape =
        (height, width) for a single image (GRAY),
        (height, width, n_channels) for a single image (RGB),
        (n_images, height, width, n_channels) for multiple images (RGB or GRAY)

    size: tuple, (wnd_h, wnd_w) window size
    stride: int, stride

    return: subwindows (torch.Tensor), shape = (n_images, n_subwindows, n_channels, wnd_h, wnd_w)
    """
    if isinstance(img, np.ndarray):
        if len(img.shape) == 4:
            img = torch.tensor(img, device=device).float().permute(0, 3, 1, 2) # (n_images, n_channels, height, width)
        if len(img.shape) == 3:
            img = torch.tensor(img, device=device).float().permute(2, 0, 1) # (n_channels, height, width)
        elif len(img.shape) == 2:
            img = torch.tensor(img).float()
    
    
    wnd_h, wnd_w = size
    if len(img.shape) == 2: # single image, GRAY: (height, width)
        img = img.unsqueeze(0)

    if len(img.shape) == 3: # single image, RGB: (n_channels, height, width)
        img = img.unsqueeze(0)  # add batch dimension

    n_channels = img.shape[1]
    n_images = img.shape[0]
    height = img.shape[-2]
    width = img.shape[-1]

    # get subwindows
    subwindows = img.unfold(2, wnd_h, stride).unfold(3, wnd_w, stride) # (n_images, n_channels, n_subwindows_h, n_subwindows_w, wnd_h, wnd_w)
    subwindows = subwindows.permute(0, 2, 3, 1, 4, 5) # (n_images, n_subwindows_h, n_subwindows_w, n_channels, wnd_h, wnd_w)
    subwindows = subwindows.reshape(n_images, -1, n_channels, wnd_h, wnd_w) # (n_images, n_subwindows, n_channels, wnd_h, wnd_w)

    # get subwindows' coordinates
    x = torch.arange(0, width - wnd_w + 1, stride)
    y = torch.arange(0, height - wnd_h + 1, stride)
    x, y = torch.meshgrid(x, y)
    x = x.reshape(-1)
    y = y.reshape(-1)
    coordinates = torch.stack((x, y), dim=1)  # (n_subwindows, 2)

    return subwindows, coordinates

def find_face(img: np.ndarray, 
              classifier:CascadeClassifier or AdaBoostClassifier, 
              feature_extractor:FeatureExtractor,
              window_size:Tuple=(19,19), 
              scale_dist:float=1.25, 
              max_size:int=300, 
              stride:int=10, 
              device:torch.device=None,
              verbose:bool=False,
              normalize_subwindows=False, 
              report_time=False,
              use_sklearn=False,
              sklearn_selector=None):
    """
    img: np.ndarray, shape = (height, width) (should be normalized, resized to same size, and gray)
    window_size: tuple, (wnd_h, wnd_w) window size used for training 
    scale_dist: float, scale distance between two scales
    max_size: int, max size of a face in the image
    stride: int, stride
    classifier: CascadeClassifier, classifier
    feature_extractor: FeatureExtractor, feature extractor
    device: torch.device, device

    return: face_coordinates: list of tuples, [(x1, y1, x2, y2), ...] (x1, y1) is the top left corner, (x2, y2) is the bottom right corner
    """
    # TODO resize input images to smaller sizes before calling this
    # TODO optimize get_subwindows
    # TODO use better scales, strides
    # TODO extract only needed features, calculate integral image of the whole image first to get them
    # TODO use integral image to calculate mean, std, normalize
    # TODO use cascade classifier
    # TODO: instead of resizing images: resize features
    # TODO: Multiple images at once
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    tasks = ['Getting Subwindows', 'Resizing Subwindows', 'Normalize', 'Extracting Features', 'Classifying']
    tasks_times = [0, 0, 0, 0, 0]

    if report_time:
        start = time.time()
    
    # img =  (img - img.mean()) / img.std()
    # get subwindows
    current_size = window_size[0], window_size[1]
    max_confidence = -np.inf
    region_max_conf = None
    face_coordinates = []
    while current_size[0] < max_size and current_size[1] < max_size and current_size[0] < img.shape[0] and current_size[1] < img.shape[1]:
        if verbose:
            print("current_size: ", current_size)

        # ++++++++++ get subwindows ++++++++++
        subwindows, coordinates = get_subwindows(img, current_size, stride, device)
        # get rid of n_channels dimension as it is 1
        subwindows = subwindows.squeeze(2) # (n_images, n_subwindows, wnd_h, wnd_w)
        orig_sh = subwindows.shape # (n_images, n_subwindows, n_channels, wnd_h, wnd_w)

        if report_time:
            tasks_times[0] += time.time() - start
            start = time.time()

        # ++++++++++ resize subwindows ++++++++++
        # Resize subwindows to window_size 
        # subwindows = subwindows.reshape(-1, orig_sh[-3], orig_sh[-2], orig_sh[-1]) # (n_images * n_subwindows, n_channels, wnd_h, wnd_w)
        # subwindows = F.interpolate(subwindows, size=window_size, mode='bilinear', align_corners=False) # (n_images, n_subwindows, n_channels, wnd_h, wnd_w)
        # subwindows = subwindows.reshape(orig_sh[0], orig_sh[1], orig_sh[2], window_size[0], window_size[1]) # (n_images, n_subwindows, n_channels, wnd_h, wnd_w)
        
        subwindows = subwindows.reshape(-1, orig_sh[-2], orig_sh[-1]) # (n_images * n_subwindows, wnd_h, wnd_w)
        subwindows = F.interpolate(subwindows.unsqueeze(1), size=window_size, mode='bilinear', align_corners=False) # (n_images * n_subwindows, 1, wnd_h, wnd_w)
        # unsqueeze(1) is needed because F.interpolate needs 4D tensor of shape (n_images, n_channels, height, width)
        subwindows = subwindows.squeeze(1) # (n_images * n_subwindows, wnd_h, wnd_w)     
        # Calculate mean and standard deviation of pixel values along axis 0
        # mean = subwindows.mean(dim=0, keepdim=True)
        # std = subwindows.std(dim=0, keepdim=True)

        if report_time:
            tasks_times[1] += time.time() - start
            start = time.time()

        # ++++++++++ normalize subwindows ++++++++++
        if normalize_subwindows:
          # normalize mean, std
          subwindows = (subwindows - subwindows.mean(dim=(1,2), keepdim=True)) / subwindows.std(dim=(1,2), keepdim=True)
          # should be equivalent to cv2.normalize
          subwindows = (subwindows - subwindows.amin(dim=(1,2), keepdim=True)) / (subwindows.amax(dim=(1,2), keepdim=True) - subwindows.amin(dim=(1,2), keepdim=True)) 
          # TODO i hate integers, make sure input is integers
          subwindows = (subwindows* 255).to(torch.uint8) 
        # Normalize pixel values of each image using mean and std
        # subwindows = (subwindows - mean) / std
        # for subwindow in subwindows:
        #   print(subwindow.shape)
        #   plt.imshow(subwindow, cmap='gray')
        #   plt.show()
        # subwindows = subwindows.reshape(orig_sh[0], orig_sh[1], window_size[0], window_size[1]) # (n_images, n_subwindows, wnd_h, wnd_w)   

        if report_time:
            tasks_times[2] += time.time() - start
            start = time.time()

        # ++++++++++ extract features ++++++++++

        if verbose:
            print("subwindows.shape: ", subwindows.shape)
            print("subwindows.shape: ", subwindows.shape)
            print("coordinates.shape: ", coordinates.shape)
        if use_sklearn:
            t_features = feature_extractor.extractFeatures2(subwindows, create_ii=True)
            t_f_idx_map = None
        else:
            t_f_idx_map, t_features = feature_extractor.extractFeaturesFromImage(subwindows,
                                                cascadeClassifier=classifier)
        
        
        if verbose:
            print("t_features.shape: ", t_features.shape)
            print("t_f_idx_map: ", t_f_idx_map)

        if report_time:
            tasks_times[3] += time.time() - start
            start = time.time()

        # ++++++++++ classify ++++++++++
        # Predict
        if use_sklearn:
            sklearn_t_features = sklearn_selector.transform(t_features.T)
            predictions = classifier.predict(sklearn_t_features)
        else:
            predictions = classifier.predict(t_features, t_f_idx_map)

        if verbose:
            print("predictions.shape: ", predictions.shape)
            print("predictions: ", predictions)

        # get face coordinates from coordinates, predictions. Put coordinates, size into face_coordinates
        # make 4 coordinates for each face (x1, y1, x2, y2)
        if not use_sklearn:
            tmp_conf = classifier.confidence(t_features, t_f_idx_map)
            arg_max = np.argmax(tmp_conf)
        else:
            tmp_conf = classifier.decision_function(sklearn_t_features)
            arg_max = np.argmax(tmp_conf)
        if tmp_conf[arg_max] > max_confidence:
          # region_max_conf = np.concatenate((coordinates[arg_max], coordinates[arg_max] + np.array(window_size)), axis=0)
          region_max_conf = np.concatenate((coordinates[arg_max], coordinates[arg_max] + np.array(current_size)), axis=0)
          max_confidence = tmp_conf[arg_max]

        # tmp_tuple = np.concatenate((coordinates[predictions == 1], coordinates[predictions == 1] + torch.tensor(window_size)), axis=1)
        tmp_tuple = np.concatenate((coordinates[predictions == 1], coordinates[predictions == 1] + torch.tensor(current_size)), axis=1)
        face_coordinates.append(tmp_tuple)

        if report_time:
            tasks_times[4] += time.time() - start
            start = time.time()

        if verbose:
            print("face_coordinates.shape: ", np.array(face_coordinates).shape)
            print("face_coordinates: ", np.array(face_coordinates))

        current_size = int(current_size[0] * scale_dist), int(current_size[1] * scale_dist)

    if report_time:
        return face_coordinates, region_max_conf, max_confidence, dict(zip(tasks, tasks_times))
    else:
        return face_coordinates, region_max_conf, max_confidence


class FaceDetector:
    def __init__(self,
                 classifier:CascadeClassifier or AdaBoostClassifier,
                 feature_extractor:FeatureExtractor,

                 window_size:Tuple=(19,19),
                 scale_dist:float=1.25,
                 max_size:int=300,
                 stride:int=10,

                 device:torch.device=None,

                 use_percentile:bool=True,
                 makeθ0:bool=False,

                 verbose:bool=False,
                 normalize_subwindows=False,
                 report_time=False,
                 use_sklearn=False):
        """
        classifier: CascadeClassifier, classifier
        feature_extractor: FeatureExtractor, feature extractor
        window_size: tuple, (wnd_h, wnd_w) window size used for training
        scale_dist: float, scale distance between two scales
        max_size: int, max size of a face in the image
        stride: int, stride
        device: torch.device, device
        use_percentile: bool, if used percentile to minimize number of features
        makeθ0: bool, if True, will make Σ α * hi(x) > 0 and hi(x) ϵ {-1, 1} 
        verbose: bool, verbose
        normalize_subwindows: bool, normalize subwindows
        report_time: bool, report time
        use_sklearn: bool, use sklearn's AdaBoostClassifier
        """
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.window_size = window_size
        self.scale_dist = scale_dist
        self.max_size = max_size
        self.stride = stride

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.use_percentile = use_percentile
        self.makeθ0 = makeθ0
        self.verbose = verbose
        self.normalize_subwindows = normalize_subwindows
        self.report_time = report_time
        self.use_sklearn = use_sklearn
        
        
        # save feature descriptions
        self.f_locations, self.f2_m, self.f3_m, self.f4_m = feature_extractor.clf_2_f_desc(cascadeClassifier=classifier, percentile=self.use_percentile)

        # update classifier with f_locations
        if isinstance(classifier, CascadeClassifier):
            classifier.updateIndecies(self.f_locations)

        # ∑ α * hi(x) > 0 and hi(x) ϵ {-1, 1}
        if makeθ0:
            self.classifier.changePN(1, -1)
            self.classifier.updateThreshold(0)


    def find_face(self,
                   img: np.ndarray, 
                   predict=True,
                   confidence=True):
        """
        img: np.ndarray, shape = (height, width) (should be normalized, resized to same size, and gray)
        return: face_coordinates: list of tuples, [(x1, y1, x2, y2), ...] (x1, y1) is the top left corner, (x2, y2) is the bottom right corner
        """
        # TODO resize input images to smaller sizes before calling this
        # TODO optimize get_subwindows
        # TODO use better scales, strides
        # TODO extract only needed features, calculate integral image of the whole image first to get them
        # TODO use integral image to calculate mean, std, normalize
        # TODO use cascade classifier
        # TODO: instead of resizing images: resize features
        # TODO: Multiple images at once
        
        tasks = ['Getting Subwindows', 'Resizing Subwindows', 'Normalize', 'Extracting Features', 'Classifying', 'total']
        tasks_times = [0, 0, 0, 0, 0, 0]

        if self.report_time:
            start = time.time()
        
        # get subwindows
        current_size = self.window_size[0], self.window_size[1]
        max_confidence = -np.inf
        region_max_conf = None
        face_coordinates = []


        while current_size[0] < self.max_size and current_size[1] < self.max_size and current_size[0] < img.shape[0] and current_size[1] < img.shape[1]:
            if self.verbose:
                print("current_size: ", current_size)

            # ++++++++++ get subwindows ++++++++++
            subwindows, coordinates = get_subwindows(img, current_size, self.stride, self.device)
            # get rid of n_channels dimension as it is 1
            subwindows = subwindows.squeeze(2) # (n_images, n_subwindows, wnd_h, wnd_w)
            orig_sh = subwindows.shape # (n_images, n_subwindows, n_channels, wnd_h, wnd_w)

            if self.report_time:
                tasks_times[0] += time.time() - start
                start = time.time()

            # ++++++++++ resize subwindows ++++++++++
            # Resize subwindows to window_size 
            subwindows = subwindows.reshape(-1, orig_sh[-2], orig_sh[-1]) # (n_images * n_subwindows, wnd_h, wnd_w)
            subwindows = F.interpolate(subwindows.unsqueeze(1), size=self.window_size, mode='bilinear', align_corners=False) # (n_images * n_subwindows, 1, wnd_h, wnd_w)
            
            subwindows = subwindows.squeeze(1) # (n_images * n_subwindows, wnd_h, wnd_w)     

            if self.report_time:
                tasks_times[1] += time.time() - start
                start = time.time()

            # ++++++++++ normalize subwindows ++++++++++
            if self.normalize_subwindows:
                # normalize mean, std
                subwindows = (subwindows - subwindows.mean(dim=(1,2), keepdim=True)) / subwindows.std(dim=(1,2), keepdim=True)
                # should be equivalent to cv2.normalize
                subwindows = (subwindows - subwindows.amin(dim=(1,2), keepdim=True)) / (subwindows.amax(dim=(1,2), keepdim=True) - subwindows.amin(dim=(1,2), keepdim=True)) 
                # TODO i hate integers, make sure input is integers
                subwindows = (subwindows* 255).to(torch.uint8) 

            if self.report_time:
                tasks_times[2] += time.time() - start
                start = time.time()

            # ++++++++++ extract features ++++++++++

            if self.verbose:
                print("subwindows.shape: ", subwindows.shape)
                print("subwindows.shape: ", subwindows.shape)
                print("coordinates.shape: ", coordinates.shape)
            if self.use_sklearn:
                t_features = self.feature_extractor.extractFeatures2(subwindows, create_ii=True)
                t_f_idx_map = None
            else:
                # t_f_idx_map, t_features = self.feature_extractor.extractFeaturesFromImage(subwindows,
                #                                     cascadeClassifier=self.classifier)
                ii = self.feature_extractor.getIntegralImage(subwindows).numpy()
                t_features = self.feature_extractor.getFeaturesFromDesc(self.f2_m, self.f3_m, self.f4_m, (ii, None))
            
            
            if self.verbose:
                print("t_features.shape: ", t_features.shape)
                print("t_f_idx_map: ", t_f_idx_map)

            if self.report_time:
                tasks_times[3] += time.time() - start
                start = time.time()

            # ++++++++++ classify ++++++++++
            #% Predict
            if predict:
                if self.use_sklearn:
                    predictions = self.classifier.predict(t_features.T)
                else:
                    # predictions = self.classifier.predict(t_features, t_f_idx_map)
                    predictions = self.classifier.predict2(t_features)

                if self.verbose:
                    print("predictions.shape: ", predictions.shape)
                    print("predictions: ", predictions)

            if confidence:
                #% Confidence
                # get face coordinates from coordinates, predictions. Put coordinates, size into face_coordinates
                # make 4 coordinates for each face (x1, y1, x2, y2)
                if not self.use_sklearn:
                    # tmp_conf = self.classifier.confidence(t_features, t_f_idx_map)
                    tmp_conf = self.classifier.confidence2(t_features)
                    arg_max = np.argmax(tmp_conf)
                else:
                    tmp_conf = self.classifier.decision_function(t_features.T)
                    arg_max = np.argmax(tmp_conf)
                if tmp_conf[arg_max] > max_confidence:
                    region_max_conf = np.concatenate((coordinates[arg_max], coordinates[arg_max] + np.array(current_size)), axis=0)
                    max_confidence = tmp_conf[arg_max]

            tmp_tuple = np.concatenate((coordinates[predictions == 1], coordinates[predictions == 1] + torch.tensor(current_size)), axis=1)
            face_coordinates.append(tmp_tuple)

            if self.report_time:
                tasks_times[4] += time.time() - start
                start = time.time()

            if self.verbose:
                print("face_coordinates.shape: ", np.array(face_coordinates).shape)
                print("face_coordinates: ", np.array(face_coordinates))

            current_size = int(current_size[0] * self.scale_dist), int(current_size[1] * self.scale_dist)


        if self.report_time:
            tasks_times[5] = sum(tasks_times)
            return face_coordinates, region_max_conf, max_confidence, dict(zip(tasks, tasks_times))
        else:
            return face_coordinates, region_max_conf, max_confidence