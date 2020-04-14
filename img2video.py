import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='dir for images')
args = parser.parse_args()

images = os.listdir(os.path.join(args.i))

videowriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (2560, 1440))

for i in range(len(images)):
    mat = cv2.imread(os.path.join(args.i, f'{i}.jpg'))
    videowriter.write(mat)

videowriter.release()
