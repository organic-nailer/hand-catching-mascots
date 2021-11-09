class DetectedRect:
    def __init__(self, name: str, distance: float, x0: int, x1: int, y0: int, y1:int, centerX: int, centerY: int, surfaceArea: int) -> None:
        self.name: str = name
        self.distance: float = distance
        self.x0: int = x0
        self.x1: int = x1
        self.y0: int = y0
        self.y1: int = y1
        self.centerX: int = centerX
        self.centerY: int = centerY
        self.surfaceArea: int = surfaceArea

    def get_img(self,frame):
        return frame[self.y0:self.y1,self.x0:self.x1]
