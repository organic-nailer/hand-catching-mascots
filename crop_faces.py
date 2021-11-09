import cv2
import numpy as np
import glob
import image_processing as ip

# 顔近傍の画像を切り取って保存する

data = glob.glob("./data_face/*.jpg")
n_data = len(data)

static_imgs = "C:/Users/hykwy/Downloads/Photos/*.jpg"

if static_imgs != None:
    paths = glob.glob(static_imgs)
    for path in paths:
        img = cv2.imread(path)
        rects = ip.get_skin_areas(img)
        face = rects[0]
        cv2.imwrite(
                f"./data_face/{n_data}.jpg", 
                img[face.y0:face.y1,face.x0:face.x1]
            )
        n_data+=1
else:

    cap = cv2.VideoCapture(0)

    while True:
        # 画像取得
        ret, frame = cap.read()
        cv2.imshow("camera", frame)
    
        cv2.imshow("skin", ip.get_skin_mask(frame))
        rects = ip.get_skin_areas(frame)
        if len(rects) < 2:
            continue
        face = rects[1]
    
        k = cv2.waitKey(1)
        if k == ord("c"):
            cv2.imwrite(
                    f"./data_face/{n_data}.jpg", 
                    frame[face.y0:face.y1,face.x0:face.x1]
                )
            n_data+=1
        cv2.rectangle(frame, (face.x0, face.y0), (face.x1, face.y1), (0, 0, 255), 5)
        cv2.imshow("rect", frame)
    
        # 終了処理
        if k == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()