#   use this reproject.py with input intrinsic and extrinsic to get reproject error

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
dist_coeffs=[ 2.04433223e-01, -8.38538735e-01,  2.67934454e-06, -2.45337372e-04,  9.55365969e-01]
rvecs=[-0.08797623, 0.01320572, -0.01609278]
tvecs=[-22.56968043, -54.01282708, 156.95175402]

camera_matrix = np.array(camera_matrix,dtype=np.float32).reshape(3,3)
dist_coeffs = np.array(dist_coeffs,dtype=np.float32).reshape(1,5)
rvecs = np.array(rvecs,dtype=np.float32).reshape(3,1)
tvecs = np.array(tvecs,dtype=np.float32).reshape(3,1)


# 计算重投影误差
error_list = []
mean_error = 0
for i in range(len(world_points)):
    img_points_reproj, _ = cv2.projectPoints(world_points[i], rvecs, tvecs, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
    error_list.append(error)
    mean_error += error
mean_error /= len(world_points)



# === 保存标定结果 ===
def save_calibration_results(filename, camera_matrix, dist_coeffs, rvec, tvec, mean_error):
    with open(filename, 'w') as f:
        f.write(f"=== Camera Calibration Results (Generated on {datetime.now()}) ===\n\n")
        f.write(f"Chessboard Square Size: {square_size} mm\n")
        f.write(f"Chessboard Dimensions: {num_x} x {num_y} (inner corners)\n")
        f.write(f"Mean Reprojection Error: {mean_error:.4f} pixels\n")
        f.write(f"Error list: \n")
        np.savetxt(f,np.array(error_list).reshape(1,-1),fmt='%.4f')

        f.write("\nCamera Matrix (Intrinsic):\n")
        np.savetxt(f, camera_matrix, fmt='%.8f')

        f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%.8f')

        f.write("\nRotation Vector (rvec) for the first image:\n")
        np.savetxt(f, rvec, fmt='%.8f')

        f.write("\nTranslation Vector (tvec) for the first image (in mm):\n")
        np.savetxt(f, tvec, fmt='%.8f')


save_calibration_results(
    filename="test_result.txt",
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvec=rvecs,
    tvec=tvecs,
    mean_error=mean_error
)

print("Testing completed! Results saved to 'test_result.txt'")
