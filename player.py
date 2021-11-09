from util import calc_distance

class Player:
    def __init__(self, name, pref: str) -> None:
        self.name = name
        self.facePositions = []
        self.faceX = None
        self.faceY = None
        self.handPositions = []
        self.handX = None
        self.handY = None
        self.handState = None # None or rock or paper
        self.handClenched = False
        self.score: int = 0
        self.pref = pref
        self.pastSurfaceArea = 1
        self.rockedCount = 0

    def setFacePosition(self, x, y):
        if len(self.facePositions) < 5:
            self.facePositions.append([x,y])
            self.faceX = x
            self.faceY = y
            return
        if len(self.facePositions) == 6:
            del(self.facePositions[0])
        nearCount = 0
        for past in self.facePositions:
            d = calc_distance(past[0], past[1], x, y)
            if d < 500:
                nearCount += 1
        if nearCount >= 3:
            self.faceX = x
            self.faceY = y
        self.facePositions.append([x,y])

    def setHandPosition(self, x, y) -> bool:
        if len(self.handPositions) < 5:
            self.handPositions.append([x,y])
            self.handX = x
            self.handY = y
            return True
        if len(self.handPositions) == 6:
            del(self.handPositions[0])
        nearCount = 0
        for past in self.handPositions:
            d = calc_distance(past[0], past[1], x, y)
            if d < 500:
                nearCount += 1
        if nearCount >= 3:
            self.handX = x
            self.handY = y
            self.handPositions.append([x,y])
            return True
        self.handPositions.append([x,y])
        return False

    def updateHandArea(self, hand):
        state = self.handdet(hand)
        self.handState = state
        if state == "rock":
            self.rockedCount = 10
            self.handClenched = True
        elif self.rockedCount > 0:
            self.rockedCount -= 1
            self.handClenched = True
        else:
            self.handClenched = False
        self.pastSurfaceArea = hand.surfaceArea


    def handdet(self, hand):
        if ((hand.surfaceArea - self.pastSurfaceArea) / self.pastSurfaceArea)>0.1:
            return "paper"
        elif((hand.surfaceArea - self.pastSurfaceArea) / self.pastSurfaceArea)<-0.1:
            return "rock"
        else:
            return None

    def updateScore(self, add: int) -> None:
        self.score += add
