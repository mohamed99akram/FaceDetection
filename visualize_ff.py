import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from detect_face import MinFaceDetector
import torch
import tqdm
import glob
from random import shuffle
import argparse
import os

dsets = [
    '../FDDB/originalPics/*/*/*/*/*.jpg',
    '../SBI2/data/FaceForensics++/original_sequences/youtube/c23/frames/*/*.png',
    '../FF++/full_frames/*.png',
    os.path.expanduser('~/Pictures/Screenshots/Screenshot from 2023-07-0*')
]
def show(args):
    face_detector:MinFaceDetector = joblib.load(args.face_detector)
    face_detector.setup_device()
    
    face_detector.min_size=args.min_size if args.min_size>0 else face_detector.min_size
    face_detector.max_size=args.max_size if args.max_size>0 else face_detector.max_size
    face_detector.stride=args.stride if args.stride>0 else face_detector.stride
    face_detector.scale_dist=args.scale_dist if args.scale_dist>0 else face_detector.scale_dist

    images_path = dsets[args.dataset_index]
    image_names = glob.glob(images_path)
    if len(image_names) == 0:
        print('No images with this glob')
        exit()
    
    if bool(args.shuffle):
        shuffle(image_names)

    for j in range(0, min(args.n_images, len(image_names))):
        # # show rectangle
        img_path = image_names[j]
        img_name = img_path.split('/')[-1]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (int(img.shape[1] * args.resize_factor), int(img.shape[0] * args.resize_factor)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_coordinates, region, _, time= face_detector.find_face(gray)

        print('Total time: ', time['total'])
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show face detection')

    parser.add_argument('--min_size', type=int, default=-1, help='min_size')
    parser.add_argument('--max_size', type=int, default=-1, help='max_size')
    parser.add_argument('--stride', type=int, default=-1, help='stride')
    parser.add_argument('--scale_dist', type=float, default=-1, help='scale_dist')
    parser.add_argument('-di', '--dataset_index', type=int, default=0, help=f'dataset index, one of {list(range(len(dsets)))}')
    parser.add_argument('-rf', '--resize_factor', type=float, default=1, help='resize_factor')
    parser.add_argument('-n', '--n_images', type=int, default=50, help='number of images')
    parser.add_argument('-fd', '--face_detector', type=str, default='models/faceDetector16_50_100_150_200.joblib', help='face detector path')
    parser.add_argument('-sh', '--shuffle', type=int, default=1, help='shuffle images')

    args = parser.parse_args()
    show(args)

# python3 visualize_ff.py -di 2 -rf 0.5 -sh 0 -n 10 -fd models/faceDetector.joblib