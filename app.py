import cv2
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

# 2002/08/11/big/img_591
# 1
# 123.583300 85.549500 1.265839 269.693400 161.781200  1

img = cv2.imread('img_591.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

center = (269, 161)
dx = 85
dy = 123

img = cv2.circle(img, center, 5, (255,0,0), -1)
p1 = (center[0] - dx, center[1] - dy)
p2 = (center[0] + dx, center[1] + dy)
img = cv2.rectangle(img, p1, p2, (0,255, 0), 2)
imshow(img);plt.show()
