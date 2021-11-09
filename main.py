from game import Game
import cv2

# メイン関数
def main():
    print("Hello, world!")
    game = Game()
    game.state = "detection"
    while True:
        game.update()
        # 終了オプション
        k = cv2.waitKey(10) # 30msキー入力を待つ
        if k == ord("q"):
            break
        if k == ord("z"): # [z]を押したらゲーム開始
            game.consumeState()
        if k == ord("f"):
            game.finishGame()
        if k == ord("a"):
            game.capture()
        # ある方向にボールを自動で動かす
        if k== ord("c"):
            game.addBall()
        if k == ord("s"):
            game.skipToGame()
    game.dispose()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
