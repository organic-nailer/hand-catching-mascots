import glob

import cv2
import numpy as np
from skimage.feature import hog

CLASSES = [ "A", "B", "hand"]
CLS2LABEL = { "A": 0, "B": 1, "hand": 2 }

# 特徴量とラベルを保持するリスト
features = []
labels = []

for c in CLASSES:
    label = CLS2LABEL[c]

    # 画像のパスの一覧(リスト)を取得
    image_paths = glob.glob(f"./data_face_masked/{c}/*.jpg")

    for path in image_paths:
        # 画像の読み込み
        img = cv2.imread(path)

        # grayscaleに変換
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # リサイズ (H, W) = (56, 56) ※要調整
        gray = cv2.resize(gray, (56, 56))

        # HOGによって特徴抽出
        feat = hog(gray)

        # リストに追加
        features.append(feat)
        labels.append(label)

# numpy配列に変換
features = np.array(features, dtype=np.float32)
labels = np.array(labels, dtype=np.uint8)

# 配列を保存
np.save("./data_face_masked/features.npy", features)
np.save("./data_face_masked/labels.npy", labels)
