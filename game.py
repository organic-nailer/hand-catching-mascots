import cv2
import numpy as np
from face_detector import FaceDetector, get_static_detector
import image_processing as ip
import hassya as hassya
from random import randint
from player import Player
from util import calc_distance
from detect_hand import detect_hand
import time

WINDOW_WIDTH = 960
WINDOW_HEIGHT = 720
GAME_NAME = "Hand Catching Mascots"

startBackground = cv2.resize(cv2.imread("image_data/title.png"), (WINDOW_WIDTH, WINDOW_HEIGHT))
finishedBackground = np.full((WINDOW_HEIGHT, WINDOW_WIDTH, 3), [255, 250, 251], dtype=np.uint8)
cv2.rectangle(finishedBackground, (0, int(WINDOW_HEIGHT * 0.2)), (int(WINDOW_WIDTH * 0.6), int(WINDOW_HEIGHT * 0.8)), (255, 153, 135)[::-1], -1)
cv2.circle(finishedBackground,(int(WINDOW_WIDTH * 0.6),WINDOW_HEIGHT // 2), int(WINDOW_HEIGHT * 0.3), (255, 153, 135)[::-1], -1)

COLOR_SAITAMA = (230, 54, 10)[::-1]
COLOR_KOUCHI = (122, 23, 28)[::-1]
COLOR_TEXT_ORANGE = (136, 45, 63)[::-1]
COLOR_YELLOW = (254, 226, 191)[::-1]

class Game:
    cap = cv2.VideoCapture(0)

    # ゲームを作ったときに初期化するために呼び出される関数
    def __init__(self) -> None:
        # ゲームの現在の状態
        # ["beforeStart","detection","pendingStart","inGame","finished"]
        self.state = "beforeStart"
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.countDown = "?"
        self.balls = []
        self.frameHeight = -1
        self.frameWidth = -1
        self.playerA = Player("A", "Saitama")
        self.playerB = Player("B", "Kouchi")
        self.capturedA = []
        self.capturedB = []
        self.detectionBufferLeft = None
        self.detectionBufferRight = None
        self.faceDetector = FaceDetector()
        self.detectionFlag = False
        self.gameTick = 0
        self.debug = False
        self.finishedFrame = finishedBackground
        self.gameStartTime = 0

    # ゲームを消すときに安全のため呼び出す
    def dispose(self):
        self.cap.release()

    # 毎フレーム呼ばれる
    # フレームの計算、画面の更新を行う
    def update(self):
        ret, frame = self.cap.read()
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        frame = cv2.flip(frame,1)
        self.frameHeight = frame.shape[0]
        self.frameWidth = frame.shape[1]
        ############## この間に画面を更新処理を書く start ###################
        if self.state == "beforeStart": # スタート画面
            frame = startBackground
        elif self.state == "detection": # 顔登録画面
            skin_mask = ip.get_skin_mask(frame)
            if self.debug:
                cv2.imshow("skin", skin_mask)
            skin_areas = ip.get_skin_areas(frame)
            l_center_x = self.frameWidth // 4
            r_center_x = self.frameWidth * 3 // 4
            center_y = self.frameHeight * 0.395
            left_face = next(filter(lambda area: 
                area.x0 <= l_center_x <= area.x1 and 
                area.y0 <= center_y <= area.y1, 
                skin_areas), None)
            if left_face is not None:
                self.detectionBufferLeft = left_face.get_img(frame)
            else:
                self.detectionBufferLeft = None
            right_face = next(filter(lambda area: 
                area.x0 <= r_center_x <= area.x1 and 
                area.y0 <= center_y <= area.y1, 
                skin_areas), None)
            if right_face is not None:
                self.detectionBufferRight = right_face.get_img(frame)
            else:
                self.detectionBufferRight = None
            darked = ip.darker(frame,0.5)
            x0 = self.frameWidth * 1 // 8
            x1 = self.frameWidth * 3 // 8
            y0 = self.frameHeight // 6
            y1 = self.frameHeight * 5 // 8
            darked[y0:y1,x0:x1] = frame[y0:y1,x0:x1]
            self.drawText(darked, x0, y1+30, "SAITAMA", COLOR_SAITAMA)
            if left_face != None:
                cv2.rectangle(darked, (x0,y0), (x1,y1), COLOR_SAITAMA, 5)
            x0 = self.frameWidth * 5 // 8
            x1 = self.frameWidth * 7 // 8
            darked[y0:y1,x0:x1] = frame[y0:y1,x0:x1]
            self.drawText(darked, x0,y1+30, "KOUCHI", COLOR_KOUCHI)
            if right_face != None:
                cv2.rectangle(darked, (x0,y0), (x1,y1), COLOR_KOUCHI, 5)
            frame = darked
            if self.detectionFlag:
                self.gameTick += 1
                if self.gameTick % 10 == 0: # 10フレーム毎
                    self.capture()
                self.drawText(frame, self.frameWidth // 2, self.frameHeight * 7 // 8, str(len(self.capturedA)))
                if len(self.capturedA) > 10:
                    self.drawText(frame, self.frameWidth // 2 - 150, self.frameHeight * 7 // 8 + 40, "press z to start")
            else:
                self.drawText(frame, self.frameWidth // 2 - 150, self.frameHeight * 7 // 8, "press z to capture")
        elif self.state == "pendingStart":
            frame = ip.darker(frame, 0.5)
            self.drawText(frame, self.frameWidth // 2 - 180, self.frameHeight // 2 + 200, self.countDown, COLOR_YELLOW, 40)
        elif self.state == "inGame":
            self.gameTick += 1
            if self.gameTick > 600:
               self.finishGame()
            if self.debug:
                skin_mask = ip.get_skin_mask(frame)
                cv2.imshow("skin", skin_mask)
            faces, hands = self.faceDetector.get_faces_and_hands(frame, debug=self.debug)
            frame = self.updatePlayer(frame, faces, hands)
            if self.gameTick % 5 == 0:
                # ランダムでボールを発射する
                self.addBall()
            self.updateAndDrawBalls(frame)
            self.drawText(frame, self.frameWidth - 250, 50, "Time:" + str((600 - self.gameTick)), size=3)
        elif self.state == "finished":
            frame = self.finishedFrame
        else:
            print("unknown state::",self.state)
        ############## この間に画面を更新処理を書く end   ###################
        if self.debug:
            self.drawText(frame, 20, 20 + 50, self.state)
        cv2.imshow("camera", frame)

    def capture(self):
        if self.state != "detection":
            return
        if self.detectionBufferLeft is None or self.detectionBufferRight is None:
            print("写ってないよ")
            return
        if self.debug:
            cv2.imshow("capturedA:" + str(len(self.capturedA)), self.detectionBufferLeft)
            cv2.imshow("capturedB:" + str(len(self.capturedB)), self.detectionBufferRight)
        self.capturedA.append(self.detectionBufferLeft)
        self.capturedB.append(self.detectionBufferRight)

    def updatePlayer(self, frame, faces, hands):
        if faces[0] != None:
            self.playerA.setFacePosition(faces[0].centerX, faces[0].centerY)
        if faces[1] != None:
            self.playerB.setFacePosition(faces[1].centerX, faces[1].centerY)
        if hands[0] != None:
            updated = self.playerA.setHandPosition(hands[0].centerX, hands[0].centerY)
            if updated:
                self.playerA.updateHandArea(hands[0])
                roi = hands[0].get_img(frame)
                hand_mask = ip.get_skin_mask(roi)
                roi[hand_mask == 255] = COLOR_SAITAMA
        else:
            self.playerA.handClenched = False
        if hands[1] != None:
            updated = self.playerB.setHandPosition(hands[1].centerX, hands[1].centerY)
            if updated:
                self.playerB.updateHandArea(hands[1])
                roi = hands[1].get_img(frame)
                hand_mask = ip.get_skin_mask(roi)
                roi[hand_mask == 255] = COLOR_KOUCHI
        else:
            self.playerB.handClenched = False            
        
        if self.debug:
            if self.playerA.faceX != None:
                frame = cv2.circle(frame,(self.playerA.faceX,self.playerA.faceY), 20, (0,0,200), 3)
            if self.playerB.faceX != None:
                frame = cv2.circle(frame,(self.playerB.faceX,self.playerB.faceY), 20, (0,200,0), 3)
            if self.playerA.handX != None:
                frame = cv2.circle(frame,(self.playerA.handX,self.playerA.handY), 20, (0,200,200), 3)
            if self.playerB.handX != None:
                frame = cv2.circle(frame,(self.playerB.handX,self.playerB.handY), 20, (200,200,0), 3)

        self.drawText(frame, 5, 100, "SAITAMA:" + str(self.playerA.score) + ( " close " + str(self.playerA.handClenched) if self.debug else "" ), COLOR_SAITAMA, 3)
        self.drawText(frame, 5, 150, "KOUCHI:" + str(self.playerB.score) + (" close " + str(self.playerB.handClenched) if self.debug else "" ), COLOR_KOUCHI, 3)

        return frame

    def updateAndDrawBalls(self,outFrame):
        #消した画像を格納するリストを作成
        removeScheduled = []
        xLimit = outFrame.shape[1]
        yLimit = outFrame.shape[0]

        #print("handA:",self.playerA.handX, " ", self.playerA.handY)

        # ボールの再配置
        for i in range(len(self.balls)):
            ball = self.balls[i]
            #print("ball:", ball.x, " ", ball.y)
            if ball.removed==False:
                # 手との衝突判定
                # 自分のと同じやつかつ手を閉じる動作をしたときにゲット
                if ball.label == "A" \
                    and self.playerA.handX != None \
                    and self.playerA.handClenched \
                    and calc_distance(ball.x,ball.y,self.playerA.handX, self.playerA.handY) < 40**2:
                    ball.removed = True
                    self.playerA.updateScore(10)
                    removeScheduled.append(ball)
                    continue
                if ball.label == "B" \
                    and self.playerB.handX != None \
                    and self.playerB.handClenched \
                    and calc_distance(ball.x,ball.y,self.playerB.handX, self.playerB.handY) < 40**2:
                    ball.removed = True
                    self.playerB.updateScore(10)
                    removeScheduled.append(ball)
                    continue
                # region of interest (注目領域)
                roi = outFrame[(ball.y -  hassya.IMAGE_H) : (ball.y + hassya.IMAGE_H), (ball.x - hassya.IMAGE_W) : (ball.x + hassya.IMAGE_W)]
                
                # 注目領域内にある，埋め込みたい画像の部分を黒塗りする
                frame_bg = cv2.bitwise_and(roi, roi, mask=ball.mask)

                # 黒塗りした部分に，画像を埋め込む
                dst = cv2.add(frame_bg, ball.img)
                
                outFrame[(ball.y -  hassya.IMAGE_H) : (ball.y + hassya.IMAGE_H), (ball.x - hassya.IMAGE_W) : (ball.x + hassya.IMAGE_W)] = dst
                    
                # ある方向にボールを自動で動かす
                ball.x+=ball.speed_x
                ball.y+=ball.speed_y
                
                # 枠外に出たら削除し、削除用のリストに格納
                if 0 > ball.x - hassya.IMAGE_W or ball.x + hassya.IMAGE_W > xLimit or 0 > ball.y - hassya.IMAGE_H or ball.y + hassya.IMAGE_W > yLimit:
                    ball.removed=True
                    removeScheduled.append(ball)
        # 消えたデータを削除        
        for i in range(len(removeScheduled)):
            self.balls.remove(removeScheduled[i])
        # 結果画像の表示
        # cv2.imshow("output", stadium_copy)

    def addBall(self):
        if self.state != "inGame":
            return
        if self.frameHeight < 0 or self.frameWidth < 0:
            return
        #0の時はball 1の時はsecond を出力
        u=randint(0,1)
        if u==1:
            ball = hassya.get_ball_saitama(self.frameWidth//2,self.frameHeight//2)
            self.balls.append(ball)
        else:
            ball = hassya.get_ball_kochi(self.frameWidth//2,self.frameHeight//2)
            self.balls.append(ball)

    # 画面にテキストを表示する
    def drawText(self, frame, x, y, text, color=(0,255,255), size=2):
        cv2.putText(
            frame,text,(x, y),
            cv2.FONT_HERSHEY_PLAIN,size,color,3,
        )

    # 状態を次に進める
    def consumeState(self):
        if self.state == "beforeStart":
            self.startDetection()
        elif self.state == "detection":
            if not self.detectionFlag:
                self.detectionFlag = True
                self.gameTick = 0
                return
            if len(self.capturedA) < 10:
                print("画像が足りないよ！")
                return
            self.faceDetector.train([
                ["A", self.capturedA],
                ["B", self.capturedB]
            ])
            self.startGame()
        elif self.state == "finished":
            self.startGame()
        else:
            print("unknown state: ", self.state)

    # Detectionフェーズを開始
    def startDetection(self):
        self.state = "detection"
        print("state updated: ", self.state)

    def skipToGame(self):
        if not self.debug:
            return
        self.faceDetector = get_static_detector(masked=True)
        self.startGame()

    # 実際のゲームをカウントダウンしてから開始
    def startGame(self):
        self.state = "pendingStart"
        print("state updated: ", self.state)
        self.countDown = "3"
        self.update()
        cv2.waitKey(1000)
        self.countDown = "2"
        self.update()
        cv2.waitKey(1000)
        self.countDown = "1"
        self.update()
        cv2.waitKey(1000)
        self.countDown = "START"
        self.update()
        cv2.waitKey(500)
        self.gameTick = 0
        # ミリ秒で保存
        self.gameStartTime = time.perf_counter_ns() // 1000000
        self.state = "inGame"
        print("state updated: ", self.state)

    # ゲームを終わらせる
    def finishGame(self):
        self.state = "finished"
        print("state updated: ", self.state)
        finishedTime = time.perf_counter_ns() // 1000000
        print("Game time:", finishedTime - self.gameStartTime, "ms")
        self.finishedFrame = finishedBackground.copy()
        self.drawText(self.finishedFrame, 20, self.frameHeight//3, "GAME OVER", COLOR_TEXT_ORANGE, 3)
        winner = "Even"
        if self.playerA.score > self.playerB.score:
            winner = "SAITAMA wins!"
        elif self.playerB.score > self.playerA.score:
            winner = "KOUCHI wins!"
        self.drawText(self.finishedFrame, 20, self.frameHeight//3 + 100, winner, COLOR_TEXT_ORANGE, 5)

        scoreText = "KOUCHI:" + str(self.playerB.score) + " SAITAMA:" + str(self.playerA.score)
        self.drawText(self.finishedFrame, 20, self.frameHeight//3 + 150, scoreText, COLOR_TEXT_ORANGE, 2)

        self.drawText(self.finishedFrame, 20, self.frameHeight//3 * 2, "Press z to restart", COLOR_TEXT_ORANGE, 2)
        self.drawText(self.finishedFrame, 20, self.frameHeight//3 * 2 + 50, "q to quit", COLOR_TEXT_ORANGE, 2)

    def toggleDebugMode(self):
        self.debug = not self.debug
        print("debugMode: ", self.debug)
