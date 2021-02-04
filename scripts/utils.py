# import cv2
# import numpy as np

# def plot_object(pos):
#     size = 1024
#     pos *= size/2
#     pos += size/2
#     img = np.zeros((size,size,3), np.uint8)
#     for p in pos:
#         cv2.circle(img, (int(p[0]), int(p[2])), 2, (255,255,255))
#     return img