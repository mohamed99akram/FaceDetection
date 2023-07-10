import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_face import FaceDetector
import torch
import tqdm
# class FaceDetectionModel:
#     '''
#     Class for the Face Detection Model.
#     '''
#     def __init__(self, classifier_path, face_detector_model_path, feature_extractor_path):
#         self.face_detector = FaceDetector(face_detector_model_path)
# load hFeatures6/faceDetector.joblib


# classifier = joblib.load('hFeatures6/cascadeClassifier.joblib')
# feature_extractor = joblib.load('hFeatures6/feature_extractor.joblib')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# faceDetector = FaceDetector(classifier, \
#                             feature_extractor,\

#                             window_size=(24, 24),\
#                             scale_dist=1.1,
#                             max_size=300,
#                             stride=5,

#                             device=device,
#                             use_percentile=False,
#                             makeθ0=False,

#                             verbose=False,
#                             normalize_subwindows=False,
#                             report_time=True,
#                             use_sklearn=False
#                             )


face_detector = joblib.load('hFeatures5_5/faceDetector.joblib')
face_detector.max_size=500
# all_classifiers = face_detector.classifier.strong_classifiers[0].weak_classifiers
resized = False
for j in range(0, 50): # images
# for j in [171, 172, 173, 183]:
# for j in [966]:
    # for i in range(1, 200, 10):
    for i in [80]: # number of classifiers
        # face_detector.classifier.strong_classifiers[0].weak_classifiers = all_classifiers[:i]
        # print(f'Taking up to {i} classifiers out of {len(all_classifiers)}')
        # face_detector.feature_extractor.batch_size = 10000
        # # show rectangle
        img_name = f'{str(j).zfill(3)}..png'
        img_path = f'/home/akram/CMP4/GP/FF++/full_frames/{img_name}'
        # img_name = f'{str(j).zfill(3)}/0.png'
        # img_path=  f'../SBI2/data/FaceForensics++/original_sequences/youtube/c23/frames/{img_name}'
        # img_path=f'/home/akram/Pictures/Screenshots/Screenshot from 2023-07-05 12-23-01.png'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_size = img.shape
        print(original_size)
        if resized:
            img2 = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            # # make gray
            gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
            # img = cv2.resize(img, (img.shape[1],img.shape[0]))
            # img = cv2.resize(img, (640, 350))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # show gray
        # plt.imshow(gray, cmap='gray')
        # plt.show()
        # # detect face
        # _, region, _, time= face_detector.find_face(gray, min_size=24)
        face_coordinates, region, _, time= face_detector.find_face(gray)
        if resized:
            region[0] = region[0] * original_size[0] / 250
            region[1] = region[1] * original_size[1] / 250
            region[2] = region[2] * original_size[0] / 250
            region[3] = region[3] * original_size[1] / 250

        print(region)
        print(time)
        # # draw rectangle
        tmp = img.copy()
        cv2.rectangle(tmp, (region[0], region[1]), (region[2], region[3]), (0, 255, 0), 2)
        # # show image
        plt.imshow(tmp)
        plt.title(img_name)
        plt.show()

        # draw face coordinates
        tmp = img.copy()
        for x1, y1, x2, y2 in face_coordinates:
            cv2.rectangle(tmp, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # # show image
        plt.imshow(tmp)
        plt.title(img_name)
        plt.show()
