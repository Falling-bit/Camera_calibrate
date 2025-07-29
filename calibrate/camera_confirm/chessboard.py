import sys
import cv2
import numpy as np

#--生成棋盘格
image = np.ones([1200,1920,3],np.uint8)*255
x_num = 15
y_num = 9 #棋盘格10*9行*列
square_pixel = 120
x0 = square_pixel
y0 = square_pixel
def drawsquare():
    flag = -1 #flag=1/-1切换格
    for i in range(x_num-1):
        flag = 0 - flag #根据棋盘比例奇偶 x为奇删除 x为偶保留
        print(i)
        for j in range(y_num-1):
            if flag>0:
                color = [0,0,0] #黑
            else:
                color = [255,255,255]
            cv2.rectangle(image,(x0 + i*square_pixel,y0 + j*square_pixel),
                          (x0 + i*square_pixel +square_pixel,y0 + j*square_pixel +square_pixel),color,-1)
            flag = 0 - flag
    cv2.imwrite('chessboard.bmp',image)

if __name__ == '__main__':
    drawsquare()

