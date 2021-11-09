import cv2
import numpy as np
from skimage.feature import hog
from sklearn.neighbors import NearestNeighbors

#特徴の読み込み
# 特徴量の読み込み
features = np.load("./data_hand/features.npy")
labels = np.load("./data_hand/labels.npy")



model = NearestNeighbors(n_neighbors=1).fit(features)
LABEL3CLS = {0:  "rock", 1: "paper"}




def detect_hand(cropped):
    if cropped.size < 10:
        return "none", "none", 0
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (56, 56))
    feat = hog(gray)
    feat = feat.reshape(1,-1)
    distances, indices = model.kneighbors(feat)
    
    label = labels[indices[0][0]]
    class_name = LABEL3CLS[label]
    return label, class_name, distances[0][0]

