
import cv2
import os
import numpy as np
import json
import sys
import tqdm

from retinaface.pre_trained_models import get_model
import torch
import argparse
import glob
import time
from datetime import datetime
class RetinfaceFaceDetector():
    """
    Face detector using RetinaFace
    """
    def __init__(self):
        super().__init__()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Get RetinaFace model
        self.model = get_model("resnet50_2020-07-20", max_size=500,device=device)
        self.model.eval()

    def detect_faces(self, frame):
        """
        Detect faces in a frame
        """
        faces = self.model.predict_jsons(frame)
        faces = [face['bbox'] for face in faces]
        faces = np.array(faces)
        return faces
    
class Dataset():
    def __init__(self,
                 dataset_pattern="dataset",
                 verbose=False):
        self.dataset_pattern = dataset_pattern

    def buildFaces(self,
                   target_size=(24, 24),
                   paddingH=5,
                   paddingW=5,
                   destination="faces.npy",
                   save_np=True):
        """
        Build dataset of faces
        """
        face_detector = RetinfaceFaceDetector()
        faces = []
        for img_path in tqdm.tqdm(glob.glob(self.dataset_pattern)):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces_in_img = face_detector.detect_faces(img)
            for face in faces_in_img:
                x1, y1, x2, y2 = face

                x1 = max(0, x1 - paddingW)
                y1 = max(0, y1 - paddingH)
                x2 = min(img.shape[1], x2 + paddingW)
                y2 = min(img.shape[0], y2 + paddingH)

                face = img[y1:y2, x1:x2]
                face = cv2.resize(face, target_size)
                face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

                if save_np:
                    faces.append(face)
                else:
                    cv2.imwrite(destination + '/' + img_path.split('/')[-1][:-4] + f"_{x1}_{y1}_{x2}_{y2}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png", face)

        if save_np:
            faces = np.array(faces)
            np.save(destination, faces)
        

    def rects_intersect(self, rect1, rect2):
        """
        Check if two rectangles intersect
        """
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        if x1 > x2 + w2 or x2 > x1 + w1 or y1 > y2 + h2 or y2 > y1 + h1:
            return False
        return True
    def buildNonFaces(self,
                      target_size=(24, 24),
                      paddingH=10,
                      paddingW=10,
                      stride=10,
                      scale=1.25,
                      min_size=50,
                      max_size=200,
                      max_per_img=1000,
                      destination="non_faces.npy",
                      save_np=True):
        """
        Build dataset of non faces
        """
        non_faces = []
        i = 0
        for img_path in tqdm.tqdm(glob.glob(self.dataset_pattern)):
            i+=1
            if i > 100:
                break
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_detector = RetinfaceFaceDetector()
            faces_in_img = face_detector.detect_faces(img)
            face_regions = []
            for face in faces_in_img:
                x1, y1, x2, y2 = face

                x1 = max(0, x1 - paddingW)
                y1 = max(0, y1 - paddingH)
                x2 = min(img.shape[1], x2 + paddingW)
                y2 = min(img.shape[0], y2 + paddingH)

                face_regions.append((x1, y1, x2, y2))

            # Loop over subwindows of the image
            for y in range(0, img.shape[0], stride):
                for x in range(0, img.shape[1], stride):
                    # Loop over scales
                    curr_size = max_size
                    while curr_size > min_size and curr_size + y < img.shape[0] and curr_size + x < img.shape[1]:
                        for x1, y1, x2, y2 in face_regions:
                            if self.rects_intersect((x, y, curr_size, curr_size), (x1, y1, x2 - x1, y2 - y1)):
                                break
                        else:
                            non_face = img[y:y+curr_size, x:x+curr_size]
                            non_face = cv2.resize(non_face, target_size)
                            non_face = cv2.cvtColor(non_face, cv2.COLOR_RGB2GRAY)
                            if save_np:
                                non_faces.append(non_face)
                            else:
                                cv2.imwrite(destination + '/' + img_path.split('/')[-1][:-4] + f"_{x}_{y}_{curr_size}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.png", non_face)
                            if len(non_faces) >= max_per_img:
                                break
                        curr_size = int(curr_size / scale)
                #     if len(non_faces) >= max_per_img:
                #         break
                # if len(non_faces) >= max_per_img:
                #     break
            # if len(non_faces) >= max_per_img:
            #     break

        if save_np:
            non_faces = np.array(non_faces)
            np.save(destination, non_faces)



if __name__ == "__main__":
    # dataset = Dataset("dataset/*.jpg")
    dataset = Dataset('../FF++/full_frames/*.png')
    # dataset.buildFaces(target_size=(24, 24),
    #                    paddingH=5,
    #                    paddingW=5,
    #                    destination="faces.npy",
    #                    save_np=True)
    dataset.buildNonFaces(target_size=(24, 24),
                            paddingH=50,
                            paddingW=50,
                            stride=30,
                            scale=1.25,
                            min_size=50,
                            max_size=500,
                            # max_per_img=100000,
                            destination="non_faces",
                            save_np=False)
    
    # parser = argparse.ArgumentParser(description='Build dataset')
    # parser.add_argument('-f', '--faces', dest='faces', nargs='+', default=None, help='python3 dataset_builder.py -f dataset/*.jpg 24 24 5 5 faces.npy 1')
    # parser.add_argument('-n', '--non-faces', dest='non_faces', nargs='+', default=None, help='python3 dataset_builder.py -n dataset/*.jpg 24 24 10 10 10 1.25 50 200 1000 non_faces.npy 1')
    # args = parser.parse_args()

    # if args.faces:
    #     dataset = Dataset(args.faces[0])
    #     dataset.buildFaces(target_size=(int(args.faces[1]), int(args.faces[2])),
    #                        paddingH=int(args.faces[3]),
    #                        paddingW=int(args.faces[4]),
    #                        destination=args.faces[5],
    #                        save_np=int(args.faces[6]) == 1)
    # if args.non_faces:
    #     dataset = Dataset(args.non_faces[0])
    #     dataset.buildNonFaces(target_size=(int(args.non_faces[1]), int(args.non_faces[2])),
    #                           paddingH=int(args.non_faces[3]),
    #                           paddingW=int(args.non_faces[4]),
    #                           stride=int(args.non_faces[5]),
    #                           scale=float(args.non_faces[6]),
    #                           min_size=int(args.non_faces[7]),
    #                           max_size=int(args.non_faces[8]),
    #                           max_per_img=int(args.non_faces[9]),
    #                           destination=args.non_faces[10],
    #                           save_np=int(args.non_faces[11]) == 1)
