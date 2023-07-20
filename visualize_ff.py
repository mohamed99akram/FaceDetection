import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_face import FaceDetector, MinFaceDetector
import torch
import tqdm
import glob
from random import shuffle

face_detector:MinFaceDetector = joblib.load('hFeatures6/faceDetector3.joblib')
face_detector.setup_device()
print(face_detector.min_size)
print(face_detector.stride)
# face_detector.min_size=24
# face_detector.max_size=300
# face_detector.stride=5
# face_detector.scale_dist=1.25
# face_detector.min_size=40
# all_classifiers = face_detector.classifier.strong_classifiers[0].weak_classifiers
fddb_path = '../FDDB/originalPics/*/*/*/*/*.jpg'
fddb_images = glob.glob(fddb_path)
print(len(fddb_images))
shuffle(fddb_images)

resized = False
for j in range(0, 50): # images
# for j in [171, 172, 173, 183]:
# for j in [966]:
    # for i in range(1, 200, 10):
    for i in [80]: # number of classifiers
        # face_detector.classifier.strong_classifiers[0].weak_classifiers = all_classifiers[:i]
        print(face_detector.classifier.strong_classifiers[0].θ)
        face_detector.classifier.strong_classifiers[0].θ = 55
        # print(f'Taking up to {i} classifiers out of {len(all_classifiers)}')
        # face_detector.feature_extractor.batch_size = 10000
        # # show rectangle
        # img_name = f'{str(j).zfill(3)}..png'
        # img_path = f'/home/akram/CMP4/GP/FF++/full_frames/{img_name}'
        img_path = fddb_images[j]
        img_name = img_path.split('/')[-1]
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
            # img=cv2.resize(img,(250,250))
            img = cv2.resize(img, (img.shape[1], img.shape[0]))
            # img = cv2.resize(img, (img.shape[1], img.shape[0]))
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
