from typing import List, Tuple
import cv2
import glob
from detected_rect import DetectedRect
import image_processing as ip
from sklearn.neighbors import NearestNeighbors
from util import calc_distance

HAND_PATH = "./data_face/hand"

# 顔認識のためのクラス
# 画像による学習、
# 画像を入力とした認識、
# 顔と手の検出を行う
class FaceDetector:
    def __init__(self, masked = False) -> None:
        self.features = []
        self.labels = []
        self.model = None
        self.masked = masked

    def train(self, data):
        """ 画像とラベルから学習する
        
        data: List[[label, List[img]]]
        """
        self.features = []
        self.labels = []
        for d in data:
            label = d[0]
            img_list = d[1]
            for img in img_list:
                hog = ip.calc_hog(img, masked=self.masked)
                self.features.append(hog)
                self.labels.append(label)
        
        image_paths = glob.glob(HAND_PATH + "/*.jpg")
        for path in image_paths:
            img = cv2.imread(path)
            hog = ip.calc_hog(img, masked=self.masked)
            self.features.append(hog)
            self.labels.append("hand")
        print(f"{len(image_paths)}このhandを読み込みました")

        self.model = NearestNeighbors(n_neighbors=1).fit(self.features)

    def detect(self, img):
        """ imgを最近傍法にかけて一番近いラベルを返す

        Returns
        ----------
        label: str
            ラベル名
        distance: int
            一番近いものからの距離
        """
        assert(self.model != None)
        feat = ip.calc_hog(img, masked=self.masked)
        feat = feat.reshape(1,-1)
        # 最近傍探索
        distances, indices = self.model.kneighbors(feat)
        i = indices[0][0]
        label = self.labels[i]
        return label, distances[0][0]

    def get_faces_and_hands(self, frame, debug: bool = False) -> Tuple[List[DetectedRect], List[DetectedRect]]:
        rects = ip.get_skin_areas(frame)
        if len(rects) == 0:
            return [None, None], [None, None]
        for rect in rects:
            name, distance = self.detect(frame[
                rect.y0:rect.y1,
                rect.x0:rect.x1
            ])
            rect.name = name
            rect.distance = distance
            #if debug:
            #    target = rect
            #    cv2.rectangle(frame, (target.x0, target.y0), (target.x1, target.y1), (0, 255, 0), 3)
            #    cv2.putText(frame, "Label:" + str(target.name) + " d=" + str(target.distance), (target.x0+10,target.y0+10), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2,)
        # return [None, None], [None, None]
        # 距離の近い順にソート
        rects = sorted(rects, key=lambda rect: rect.distance)
        # 距離の近い中で最初のAとBを抽出
        nearestA = next(filter(lambda x: x.name == "A" and x.distance < 4, rects), None)
        nearestB = next(filter(lambda x: x.name == "B" and x.distance < 4, rects), None)
        # handと認識されているかA/Bと認識されながら距離が遠いものを抽出
        hands = list(filter(lambda x: (x.name == "hand" and x.distance < 5) or (x.name != "hand" and x.distance > 4), rects))
        handA = None
        handB = None
        # handAを取得 見つからなければNone
        if nearestA != None:
            # nearestAに近い順にソート
            sortedNearA = sorted(hands, key=lambda rect: calc_distance(
                nearestA.centerX, nearestA.centerY,
                rect.centerX, rect.centerY
            ))
            if len(sortedNearA) >= 1:
                handA = sortedNearA[0]
            if debug:
                cv2.rectangle(frame, (nearestA.x0, nearestA.y0), (nearestA.x1, nearestA.y1), (0, 0, 255), 3)
                cv2.putText(frame, "Label:" + str(nearestA.name) + " d=" + str(nearestA.distance), (nearestA.x0+5,nearestA.y0+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2,)
        # handBを取得 見つからなければNone
        if nearestB != None:
            sortedNearB = sorted(hands, key=lambda rect: calc_distance(
                nearestB.centerX, nearestB.centerY,
                rect.centerX, rect.centerY
            ))
            if len(sortedNearB) >= 1:
                handB = sortedNearB[0]
            if debug:
                cv2.rectangle(frame, (nearestB.x0, nearestB.y0), (nearestB.x1, nearestB.y1), (255, 0, 0), 3)
                cv2.putText(frame, "Label:" + str(nearestB.name) + " d=" + str(nearestB.distance), (nearestB.x0+5,nearestB.y0+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2,)
        if debug and handA != None:
            cv2.rectangle(frame, (handA.x0, handA.y0), (handA.x1, handA.y1), (0, 0, 128), 5)
            cv2.putText(frame, "HandA", (handA.x0+5,handA.y0+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2,)
        if debug and handB != None:
            cv2.rectangle(frame, (handB.x0, handB.y0), (handB.x1, handB.y1), (128, 0, 0), 5)
            cv2.putText(frame, "HandB", (handB.x0+5,handB.y0+5), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2,)
        return [nearestA, nearestB], [handA, handB]
    
def get_static_detector(masked = False) -> FaceDetector:
    detector = FaceDetector(masked=masked)
    a_paths = glob.glob("./data_face/A/*.jpg")
    b_paths = glob.glob("./data_face/B/*.jpg")
    print(f"{len(a_paths)}個のAを読み込みました")
    print(f"{len(b_paths)}個のBを読み込みました")
    a_imgs = []
    b_imgs = []
    for path in a_paths:
        img = cv2.imread(path)
        a_imgs.append(img)
    for path in b_paths:
        img = cv2.imread(path)
        b_imgs.append(img)
    detector.train([
        [ "A", a_imgs ],
        [ "B", b_imgs ]
    ])
    return detector
