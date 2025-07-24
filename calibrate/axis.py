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

images = glob.glob('photo3.jpg')
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

# 定义3D坐标轴点 (X轴红色，Y轴绿色，Z轴蓝色)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

# 用PnP算法获取旋转矩阵和平移向量
_, rvecs, tvecs, inliers = cv2.solvePnPRansac(world, corners_subpixel, camera_matrix, distortion_coefficient)

# 将3D点投影到图像平面
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, camera_matrix, distortion_coefficient)

# 获取棋盘格角点作为原点
corner = tuple(corners_subpixel[0].ravel())

# 绘制坐标轴
def draw_axes(img, corner, imgpts):
    corner = tuple(map(int, corner))  # 确保 corner 是整数元组
    imgpts = imgpts.reshape(-1, 2).astype(int)  # 确保 imgpts 是整数数组
    img = cv2.line(img, corner, tuple(imgpts[0]), (0, 0, 255), 5)  # X轴(红色)
    img = cv2.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 5)  # Y轴(绿色)
    img = cv2.line(img, corner, tuple(imgpts[2]), (255, 0, 0), 5)  # Z轴(蓝色)
    return img

# 绘制坐标轴
img_with_axes = draw_axes(dst, corner, imgpts)
cv2.imwrite('axes_img.jpg', img_with_axes)

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

# 重投影
imgpts, jac = cv2.projectPoints(axis2, rvecs, tvecs, camera_matrix, distortion_coefficient)
# 3D坐标系构建
outcome_image = draw_3D_Box(img_with_axes, corners_subpixel, imgpts)
cv2.imwrite('outcome_img.jpg', outcome_image)


#---输出
def save_calibration_results(filename, camera_matrix, dist_coeffs, rvecs, tvecs, mean_error):
    """
    保存相机标定结果到 txt 文件
    :param filename: 输出文件名（如 'calibration_results_1.txt'）
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :param rvecs: 旋转向量列表
    :param tvecs: 平移向量列表
    :param mean_error: 平均重投影误差
    """
    with open(filename, 'w') as f:
        f.write("=== Camera Calibration Results ===\n\n")

        # 保存相机内参
        f.write("Camera Matrix (Intrinsic Parameters):\n")
        np.savetxt(f, camera_matrix, fmt='%.8f', header='fx, fy, cx, cy', comments='')

        # 保存畸变系数
        f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%.8f')

        # 保存旋转向量（只保存第一张图的）
        f.write("\nRotation Vector (rvec) for the first image (Rodrigues format):\n")
        np.savetxt(f, rvecs[0], fmt='%.8f')

        # 保存平移向量（只保存第一张图的）
        f.write("\nTranslation Vector (tvec) for the first image (in world units):\n")
        np.savetxt(f, tvecs[0], fmt='%.8f')

        # 保存重投影误差
        f.write(f"\nMean Reprojection Error: {mean_error:.6f} pixels\n")

# 保存标定结果到文件
save_calibration_results(
    filename="calibration_results_3.txt",
    camera_matrix=camera_matrix,
    dist_coeffs=distortion_coefficient,
    rvecs=rvecs,
    tvecs=tvecs,
    mean_error=mean_error
)
print("Calibration results saved to 'calibration_results_2.txt'")