import cv2
import numpy as np
import image_processing as ip

cap = cv2.VideoCapture(0)

pastSurfaceArea = 1

def handdet(hand):
    if ((hand.surfaceArea - pastSurfaceArea) / pastSurfaceArea)>0.2:
        return "paper"
        
    elif((hand.surfaceArea - pastSurfaceArea) / pastSurfaceArea)<-0.2:

        
        return "rock"
    
    else:
        
        return None
        
    
    
# 実行
while True:

    # Webカメラのフレーム取得
    ret, frame = cap.read()
    cv2.imshow("camera", frame)
    mask = ip.get_skin_mask(frame)
    cv2.imshow("skin",mask)
    
    rects = ip.get_skin_areas(frame)
    if len(rects) == 0:
        continue
    
    hand = rects[0]

    judge=handdet(hand)
    print(judge)
    
    #新しいデータにかきかえ    
    pastSurfaceArea = hand.surfaceArea
    
    
    
    cv2.imshow("mask", hand.get_img(frame))

    # 終了オプション
    k = cv2.waitKey(30)
    if k == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()