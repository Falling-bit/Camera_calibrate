import cv2
import numpy as np
import glob

from cv2 import undistort
from numpy.ma.core import reshape

from chessboard import x_num, y_num

criteria = (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 30 , 0.001) #迭代终止条件(  , maxcount, eps)

num_x = x_num - 2 # chessboard x_num - 1
num_y = y_num - 2 # chessboard y_num - 1
world = np.zeros((num_x*num_y,3),np.float32)
world[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1,2) #二维坐标点

world_points = []
image_points = []

images = glob.glob('photo2.jpg')
calibrated_images = []
for image in images:
    img = cv2.imread(image)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #BGR转化为灰度
    size = gray_image.shape[::-1]
    ret,corners = cv2.findChessboardCorners(gray_image,(num_x,num_y),None)
    print(ret)
    print(corners)
    if ret:
        world_points.append(world)
        corners_subpixel = cv2.cornerSubPix(gray_image,np.float32(corners),(5,5),(-1,-1), criteria)
        #寻找亚像素
        if [corners_subpixel]:
            image_points.append(corners_subpixel)
        else:
            image_points.append(corners)
        cv2.drawChessboardCorners(img,(num_x,num_y),corners,ret)
        calibrated_images.append(img)

print(len(image_points))

#输出相机内参、畸变系数、旋转矩阵、平移向量
ret,camera_matrix,distortion_coefficient,r_vector,t_vector = cv2.calibrateCamera(world_points, image_points, size, None,None)
print("camera matrix:\n", camera_matrix)
print("distortion coefficient:\n", distortion_coefficient)
print("rotation vectors:\n",r_vector)
print("translation vectors:\n", t_vector)

dc = cv2.drawChessboardCorners(img,size,corners,None)
cv2.imwrite('drawcorners_img.jpg', dc)

#获取优化后相机内参
img2 = img
h, w = img2.shape[:2] #[0][1]
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,distortion_coefficient,(w,h),1,(w,h))
print("roi:"+ str(roi))
dst = cv2.undistort(img2,camera_matrix,distortion_coefficient,None,new_camera_matrix) #图像校正
cv2.imwrite('undistort_img.jpg', dst)

#重投影误差
total_error = 0
for i in range(len(world_points)):
    image_points2, _ = cv2.projectPoints(world_points[i],r_vector[i], t_vector[i],camera_matrix, distortion_coefficient)
    error = cv2.norm(image_points[i], image_points2,cv2.NORM_L2)/ len(image_points2)
    total_error += error

mean_error = total_error / len(world_points)
print('Reprojection error:'+ str(mean_error))

#3D盒子点
axis2 = np.float32([[0,0,0] , [0,3,0] , [3,3,0] , [3,0,0], [0,0,-3] , [0,3,-3] , [3,3,-3] , [3,0,-3]])

#3D坐标系构建
def draw_3D_Box(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    #green background
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0,255,0), -3)
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 3)

    img = cv2.drawContours(img, [imgpts[4:]], -1, (0,0,255), 3)
    return img


# 用PnP算法获取旋转矩阵和平移向量
_, r_vectors, t_vectors, inliers = cv2.solvePnPRansac(world, corners_subpixel, camera_matrix, distortion_coefficient)
# 重投影
imgpts, jac = cv2.projectPoints(axis2, r_vectors, t_vectors, camera_matrix, distortion_coefficient)
# 3D坐标系构建
outcome_image = draw_3D_Box(dst, corners_subpixel, imgpts)
cv2.imwrite('outcome_img.jpg', outcome_image)
