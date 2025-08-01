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

        print(image_points[0][0])
        print(world_points[0][0])

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
dist_coeffs=[ 2.04433223e-01, -8.38538735e-01,  2.67934454e-06, -2.45337372e-04,  9.55365969e-01]
rvecs=[-0.08797623, 0.01320572, -0.01609278]
tvecs=[-22.56968043, -54.01282708, 156.95175402]

camera_matrix = np.array(camera_matrix,dtype=np.float32).reshape(3,3)
dist_coeffs = np.array(dist_coeffs,dtype=np.float32).reshape(1,5)
rvecs = np.array(rvecs,dtype=np.float32).reshape(3,1)
tvecs = np.array(tvecs,dtype=np.float32).reshape(3,1)


def project_points(object_points, rvec, tvec, camera_matrix, dist_coeffs):
    # Step 1: 旋转向量 → 旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    object_points = np.array([object_points])

    # Step 2: 转换到相机坐标系
    X_cam = R @ object_points.T + tvec.reshape(3, 1)
    X_cam = X_cam.T  # 转置为Nx3

    # Step 3: 归一化投影
    x = X_cam[:, 0] / X_cam[:, 2]
    y = X_cam[:, 1] / X_cam[:, 2]

    # Step 4: 畸变校正
    r2 = x ** 2 + y ** 2
    radial = 1 + dist_coeffs[0][0] * r2 + dist_coeffs[0][1] * r2 ** 2 + dist_coeffs[0][4] * r2 ** 3
    x_distorted = x * radial + 2 * dist_coeffs[0][2] * x * y + dist_coeffs[0][3] * (r2 + 2 * x ** 2)
    y_distorted = y * radial + dist_coeffs[0][2] * (r2 + 2 * y ** 2) + 2 * dist_coeffs[0][3] * x * y

    # Step 5: 像素坐标
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    u = fx * x_distorted + cx
    v = fy * y_distorted + cy

    return np.column_stack((u, v))

if __name__ == '__main__':
    print(world_points[0][0])
    img_point = project_points(world_points[0][0],rvecs,tvecs,camera_matrix, dist_coeffs)
    print(img_point)