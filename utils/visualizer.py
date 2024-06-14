import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import os

def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        heat_map = scores[i] * 255
        plt.close()
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ty='heat_map'
        plt.axis('off')
        plt.savefig(os.path.join(save_dir,ty, class_name + '_{}'.format(i)), bbox_inches='tight',
                            dpi=600, pad_inches=0.0)
def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    
    return x