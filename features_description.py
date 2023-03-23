import numpy as np
import torch
class Rect():
    def __init__(self, xs, ys, xe, ye):
        self.xs = xs # x start
        self.ys = ys # y start
        self.xe = xe # x end
        self.ye = ye # y end

    def __array__(self):
        return np.array([self.xs, self.ys, self.xe, self.ye])

def describe_features(shape):
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
                        rect2.append(((ij+l1[0]), (ij+l1[1])))

                    # Vertical #+
                        # -
                    if i + 2 * k <= height and j + l <= width:
                        rect2.append(((ij+l2[0]), (ij+l2[1])))

                    # @ 3 features
                    l3 = np.array([(0,0,k,l), (0,l,k,2*l), (0,2*l,k,3*l)])
                    l4 = np.array([(0,0,k,l), (k,0,2*k,l), (2*k,0,3*k,l)])
                    # Horizontal [-][+][-]
                    if i + k <= height and j + 3 * l <= width:
                        rect3.append(((ij+l3[0]), (ij+l3[1]), (ij+l3[2])))

                    # Vertical #-
                        # +
                        # -
                    if i + 3 * k <= height and j + l <= width:
                        rect3.append(((ij+l4[0]), (ij+l4[1]), (ij+l4[2])))

                    # @ 4 features
                    l5 = np.array([(0,0,k,l), (0,l,k,2*l),(k, 0, 2*k, l), (k, l, 2*k, 2*l)])
                    # [-][+]
                    # [+][-]
                    if i + 2 * k <= height and j + 2 * l <= width:
                        rect4.append(((ij+l5[0]), (ij+l5[1]), (ij+l5[2]), (ij+l5[3])))
    return np.array(rect2), np.array(rect3), np.array(rect4)

# input: tensor of gray images: shape: (l, 19, 19) or np array of gray images: shape: (l, 19, 19) or one image
def getIntegralImage(img: torch.Tensor or np.ndarray):
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 2:
            return img.cumsum(dim=0).cumsum(dim=1)
        return img.cumsum(dim=1).cumsum(dim=2)
    elif isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            return np.cumsum(np.cumsum(img, axis=0), axis=1)
        return np.cumsum(np.cumsum(img, axis=1), axis=2)