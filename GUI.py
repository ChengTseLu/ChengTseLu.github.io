import cv2
import numpy as np

class gui:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.board()
    
    # init board
    def board(self):
        self.b = 255 * np.ones((45, 600, 3), np.uint8)

    # add texts to board
    def setTxt(self, frame, mask, temp):
        # init color
        mask_color = (0,255,0) if mask == "Mask" else (0,0,255)
        temp_color = (0,255,0) if temp and temp <= 37.5  else (0,0,255)

        # copy board and resize frame
        img = self.b.copy()
        frame = cv2.resize(frame, (600,600))

        # add texts to board
        cv2.putText(img, "Mask: ", (80,31), self.font, 0.8, (0,0,0), 2)
        cv2.putText(img, str(mask), (158,31), self.font, 0.8, mask_color, 2)
        cv2.putText(img, "TEMP: ", (350,31), self.font, 0.8, (0,0,0), 2)
        cv2.putText(img, str(temp), (430,31), self.font, 0.8, temp_color, 2)

        # vertically stack board with video stream
        h_img = cv2.vconcat([img, frame])

        cv2.imshow('M202A Final Project', h_img)




if __name__ == "__main__":
    GUI = gui()
    GUI.setTxt(True, 34)
    #GUI.setTxt(True, 36.5)

