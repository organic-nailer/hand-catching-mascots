# ライブラリのインポート
import copy

import cv2
import numpy as np
from random import randint

#ボールをクラスで定義
class Ball:
    def __init__(self,x,y,image,mask,label):
        r=randint(1,3)
        s=randint(1,3)
        t=randint(0,3)
        if r == s == 0:
            r = randint(1,2)
            s = randint(1,2)

        if t==1:
            r=-r
            s=-s
        elif t==2:
            r=-r
        elif t==3:
            s=-s
            

        self.x=x
        self.y=y
        self.removed=False
        self.speed_x=r * 2
        self.speed_y=s * 2
        self.img = image
        self.mask=mask
        self.label = label

SAITAMA_IMG = cv2.imread("./image_data/cobaton.png")
KOCHI_IMG = cv2.imread("./image_data/sinjyoukun.jpg")
SAITAMA_IMG=cv2.resize(SAITAMA_IMG,(100,100))
KOCHI_IMG = cv2.resize(KOCHI_IMG, (SAITAMA_IMG.shape[1], SAITAMA_IMG.shape[0]))
IMAGE_H, IMAGE_W = SAITAMA_IMG.shape[0] // 2, SAITAMA_IMG.shape[1] // 2

#saitama_imgの背景除
cut_rect = (1, 1, IMAGE_W*2, IMAGE_H*2)

# grubcutに必要なmaskや座標を格納するための配列の準備
ball_mask = np.zeros((IMAGE_H*2, IMAGE_W*2), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


# grubcutの実行
cv2.grabCut(SAITAMA_IMG, ball_mask, cut_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# maskを用意
ball_mask = np.where((ball_mask == 2) | (ball_mask == 0), 0, 1).astype("uint8")  
#import pdb;pdb.set_trace()
# bitwise_andを用いて前景領域を抽出
ball_fg = cv2.bitwise_and(SAITAMA_IMG, SAITAMA_IMG, mask=ball_mask)

   #ボール以外を残すためにインバースを行った。
ball_mask=1-ball_mask

#kochi_imgの背景除去



cut_rect = (1, 1, IMAGE_W*2, IMAGE_H*2)

# grubcutに必要なmaskや座標を格納するための配列の準備
second_mask = np.zeros((IMAGE_H*2, IMAGE_W*2), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


# grubcutの実行
cv2.grabCut(KOCHI_IMG, second_mask, cut_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
# maskを用意
second_mask = np.where((second_mask == 2) | (second_mask == 0), 0, 1).astype("uint8")  
#import pdb;pdb.set_trace()
# bitwise_andを用いて前景領域を抽出
second_fg = cv2.bitwise_and(KOCHI_IMG, KOCHI_IMG, mask=second_mask)

#ボール以外を残すためにインバースを行った。
second_mask=1-second_mask

def get_ball_saitama(x,y):
    return Ball(x,y, ball_fg, ball_mask, "A")

def get_ball_kochi(x,y):
    return Ball(x,y, second_fg, second_mask, "B")
