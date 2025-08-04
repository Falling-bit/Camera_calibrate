import cv2
import numpy as np
import glob
from datetime import datetime

# 棋盘格参数
from chessboard import x_num, y_num

square_size = 5.0  # 每个格子的物理尺寸（单位：mm）

# 计算内部角点数（去掉最外层的角点）
num_x = x_num - 2
num_y = y_num - 2

# 生成世界坐标系下的3D点（以第一个检测到的角点为原点）
world = np.zeros((num_x * num_y, 3), np.float32)
world[:, :2] = square_size * np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)  # 物理尺寸单位（mm）

# 标定参数
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
world_points = []
image_points = []

# 检测角点
images = glob.glob(r'./r&tvec/*.jpg')  # 替换为你的图片路径
for image in images:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

    if ret:
        # 亚像素优化
        corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        world_points.append(world)
        image_points.append(corners_subpix)


        # 可视化角点
        cv2.drawChessboardCorners(img, (num_x, num_y), corners_subpix, ret)
        cv2.imwrite('corners_detected.jpg', img)

'''
# 相机标定----
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    world_points, image_points, gray.shape[::-1], None, None
)
'''

#输入测试参数 内参+外参
camera_matrix=[[1.21913761e+03, 0.00000000e+00, 8.51803356e+02 ],
               [0.00000000e+00, 1.21960559e+03, 6.42361708e+02],
               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
dist_coeffs=[0., 0., 0., 0., 0.]

rvecs=[-0.08987502, 0.00418314, -0.01795928]
tvecs=[-26.59794645, -49.63130444, 134.50097077]

camera_matrix = np.array(camera_matrix,dtype=np.float32).reshape(3,3)
dist_coeffs = np.array(dist_coeffs,dtype=np.float32).reshape(1,5)
rvecs = np.array(rvecs,dtype=np.float32).reshape(3,1)
tvecs = np.array(tvecs,dtype=np.float32).reshape(3,1)

#求解尺度因子s方程
# 初始化矩阵 A 和 B
A = []
B = []

# 旋转矩阵
R, _ = cv2.Rodrigues(rvecs)
print(R)

# 遍历所有点
for i in range(len(world_points[0])):
    X_w, Y_w, _ = world_points[0][i]  # Z_w = 0
    u, v = image_points[0][i][0]

    # 构造方程 1: A1 * s = B1
    A1 = u
    B1 = (camera_matrix[0, 0] * R[0, 0] + camera_matrix[0, 2] * R[2, 0]) * X_w + \
         (camera_matrix[0, 0] * R[0, 1] + camera_matrix[0, 2] * R[2, 1]) * Y_w + \
         camera_matrix[0, 0] * tvecs[0] + camera_matrix[0, 2] * tvecs[2]

    # 构造方程 2: A2 * s = B2
    A2 = v
    B2 = (camera_matrix[1, 1] * R[1, 0] + camera_matrix[1, 2] * R[2, 0]) * X_w + \
         (camera_matrix[1, 1] * R[1, 1] + camera_matrix[1, 2] * R[2, 1]) * Y_w + \
         camera_matrix[1, 1] * tvecs[1] + camera_matrix[1, 2] * tvecs[2]

    A.append(A1)
    A.append(A2)
    B.append(B1)
    B.append(B2)

# 最小二乘求解 s
A = np.array(A).reshape(-1, 1)
B = np.array(B)
s = np.linalg.lstsq(A, B, rcond=None)[0][0]

print(f"最小二乘估计的尺度因子 s: {s}")

# 验证尺度因子s-----------------------------------------------------------------
reprojected, _ = cv2.projectPoints(
    np.array([[X_w, Y_w, 0]], dtype=np.float32),  # 使用已知的世界坐标
    rvecs,
    tvecs,
    camera_matrix,
    dist_coeffs
)
reprojected = reprojected.reshape(-1)
print("\n重投影验证（3d-2d）：")
print("原始图像坐标:", [u, v])
print("重投影图像坐标:", reprojected)

# 计算误差
error_pixel = np.linalg.norm(np.array([u, v]) - reprojected)
print("重投影误差(像素):", error_pixel)
#--------------------------------------------------------------------
#total error
total_error = 0
for i in range(len(world_points[0])):
    X_w, Y_w, _ = world_points[0][i]
    u, v = image_points[0][i][0]

    reprojected, _ = cv2.projectPoints(
        np.array([[X_w, Y_w, 0]], dtype=np.float32),
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs
    )
    error = np.linalg.norm(np.array([u, v]) - reprojected.reshape(-1))
    total_error += error

print("平均重投影误差:", total_error / len(world_points[0]))


#-------------------------------------------------------------------



if __name__ == '__main__':
    # 选择第一个点进行测试
    test_point_idx = 0
    X_w, Y_w, _ = world_points[0][i]  # Z_w = 0
    u, v = image_points[0][i][0]

    # 正确获取单个图像点坐标
    u, v = image_points[0][test_point_idx][0]  # 获取(u,v)坐标
    np.set_printoptions(precision=4,suppress=True)
    print('Input image point:', (u, v))

    # 将图像点转换为齐次坐标 (3x1)
    image_point = np.array([[u], [v], [1.0]], dtype=np.float32)

    # 计算相机坐标系下的点
    cam_point = np.linalg.inv(camera_matrix) @ image_point * s

    # 转换到世界坐标系
    R, _ = cv2.Rodrigues(rvecs)
    world_coord = np.linalg.inv(R) @ (cam_point - tvecs.reshape(3, 1))

    print('Output world point:', world_coord.flatten())
    print(X_w,Y_w)

    # Calculate reconstruction error
    original_point = np.array([X_w, Y_w, 0])
    error = np.linalg.norm(world_coord - original_point)
    print('Reconstruction error (mm):', error)