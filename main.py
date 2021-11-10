from game import Game
import cv2

# メイン関数
def main():
    game = Game()
    while True:
        game.update()
        # 終了オプション
        k = cv2.waitKey(10) # 10msキー入力を待つ
        if k == ord("q"):
            break
        if k == ord("z"): # [z]を押したらゲーム開始
            game.consumeState()
        if k == ord("f"): # 強制ゲーム終了
            game.finishGame()
        # ある方向にボールを自動で動かす
        if k== ord("c"):
            game.addBall()
        if k == ord("s"):
            game.skipToGame()
        if k == ord("g"):
            game.toggleDebugMode()
    game.dispose()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
