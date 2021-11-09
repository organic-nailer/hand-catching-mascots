from typing import List
import cv2
import numpy as np
from skimage.feature import hog

from detected_rect import DetectedRect

# inputに入るBGR画像をvalue倍だけ暗くする
def darker(input, value):
    img_hsv = cv2.cvtColor(input,cv2.COLOR_BGR2HSV)
    img_hsv[:,:,(2)] = img_hsv[:,:,(2)]*value
    return cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)

# マスクの白い部分だけ残して黒く塗りつぶす
# maskとtargetの縦横の大きさは一致している必要がある
def fillBGWithMask(mask, target):
    black = np.zeros_like(target)
    maskColored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return np.where(maskColored[:,:,:] == 0, black, target)

# ノイズ除去のためのカーネルの定義
kernel = np.ones((5, 5), np.uint8)

# 肌のマスクを得る
def get_skin_mask(frame):
    # 画像をRGBからHSVに変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSVによる上限、下限の設定　 ([Hue, Saturation, Value])
    hsvLower = np.array([0, 50, 60])  # 下限
    hsvUpper = np.array([30, 200, 240])  # 上限

    # HSVからマスクを作成
    hsv_mask = cv2.inRange(hsv, hsvLower, hsvUpper)
    hsv_mask = cv2.erode(hsv_mask, kernel)
    #hsv_mask = cv2.erode(hsv_mask, kernel)
    hsv_mask = cv2.dilate(hsv_mask, kernel)
    hsv_mask = cv2.dilate(hsv_mask, kernel)
    hsv_mask = cv2.erode(hsv_mask, kernel)
    #hsv_mask = cv2.dilate(hsv_mask, kernel)
    
    return hsv_mask

def get_skin_areas(frame) -> List[DetectedRect]:
    skin_mask = get_skin_mask(frame)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_mask)
    xLimit = frame.shape[1]
    yLimit = frame.shape[0]
    if nlabels >= 2:
        topIndices = np.argsort(-stats[:, 4])[1:]
        rects: List[DetectedRect] = []
        for i in topIndices:
            # 領域の外接矩形の角座標を入手
            width = stats[i, 2]
            height = stats[i, 3]
            if width <= 50 or height <= 50: # 小さければ無視
                continue
            if stats[i,4] < width * height * 0.3: # 中身が詰まってないものは多分違うので排除
                continue
            # 検知場所の矩形を取得
            x0 = stats[i, 0]
            y0 = stats[i, 1]
            x1 = x0 + width
            y1 = y0 + height
            # 顔全体を映すために枠を拡張
            x0 -= int(width * 0.1)
            y0 -= int(height * 0.3)
            x1 += int(width * 0.1)
            centerX = int(centroids[i,0])
            centerY = int(centroids[i,1])
            surface = int(stats[i,4])
            rects.append(DetectedRect(None,None,max(x0,0), min(x1, xLimit), max(y0,0), min(y1, yLimit), centerX, centerY,surface))
        return rects
    return []



def calc_hog(img, masked = False):
    if masked:
        # マスク以外の部分を埋める
        skin_mask = get_skin_mask(img)
        img = fillBGWithMask(skin_mask, img)

    # grayscaleに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # リサイズ (H, W) = (56, 56) ※要調整
    gray = cv2.resize(gray, (56, 56))
    # HOGによって特徴抽出
    feat = hog(gray)
    return feat
