
import cv2
import numpy as np
import glob
from datetime import datetime

from calibrate.axis import distortion_coefficient

# 棋盘格参数
from chessboard import x_num, y_num
from extrinsic_confirm import camera_matrix,dist_coeffs,avr_rvecs,avr_tvecs

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
images = glob.glob('photo2.jpg')  # 替换为你的图片路径
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
# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    world_points, image_points, gray.shape[::-1], None, None
)

# 计算重投影误差
mean_error = 0
for i in range(len(world_points)):
    img_points_reproj, _ = cv2.projectPoints(world_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)
    mean_error += error
mean_error /= len(world_points)
'''
# === 以第一个角点为原点绘制3D坐标系 ===
# 定义坐标系轴（X/Y轴沿棋盘格边缘，Z轴垂直向外，单位：mm）
axis_length = 20.0  # 坐标系轴长度（mm）
axis_points = np.float32([
    [axis_length, 0, 0],  # X轴（红色）
    [0, axis_length, 0],  # Y轴（绿色）
    [0, 0, -axis_length]  # Z轴（蓝色，负值表示指向相机外）
]).reshape(-1, 3)

# 使用PnP求解当前棋盘格的位姿（以第一个角点为原点）
ret, avr_rvecs, avr_tvecs = cv2.solvePnP(world, image_points[0], camera_matrix, dist_coeffs)

# 将3D轴点投影到图像平面
img_pts, _ = cv2.projectPoints(axis_points, avr_rvecs, avr_tvecs, camera_matrix, dist_coeffs)
img_pts = img_pts.astype(int).reshape(-1, 2)

# 获取第一个角点（原点）的像素坐标
origin = tuple(image_points[0][0].ravel().astype(int))

# 在图像上绘制坐标系
img_axes = cv2.imread('photo3.jpg')
cv2.line(img_axes, origin, tuple(img_pts[0]), (0, 0, 255), 3)  # X轴（红色）
cv2.line(img_axes, origin, tuple(img_pts[1]), (0, 255, 0), 3)  # Y轴（绿色）
cv2.line(img_axes, origin, tuple(img_pts[2]), (255, 0, 0), 3)  # Z轴（蓝色）
cv2.imwrite('3d_axes.jpg', img_axes)


# === 保存标定结果 ===
def save_calibration_results(filename, camera_matrix, dist_coeffs, rvec, tvec, mean_error):
    with open(filename, 'w') as f:
        f.write(f"=== Camera Calibration Results (Generated on {datetime.now()}) ===\n\n")
        f.write(f"Chessboard Square Size: {square_size} mm\n")
        f.write(f"Chessboard Dimensions: {num_x} x {num_y} (inner corners)\n")
        f.write(f"Mean Reprojection Error: {mean_error:.4f} pixels\n\n")

        f.write("Camera Matrix (Intrinsic):\n")
        np.savetxt(f, camera_matrix, fmt='%.8f')

        f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
        np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%.8f')

        f.write("\nRotation Vector (rvec) for the first image:\n")
        np.savetxt(f, rvec, fmt='%.8f')

        f.write("\nTranslation Vector (tvec) for the first image (in mm):\n")
        np.savetxt(f, tvec, fmt='%.8f')


save_calibration_results(
    filename="calibration_results_" + str(images) + ".txt",
    camera_matrix=camera_matrix,
    dist_coeffs=distortion_coefficient,
    rvec=rvec,
    tvec=tvec,
    mean_error=mean_error
)

print("Calibration completed! Results saved to 'calibration_results.txt'")
print(f"3D axes visualization saved to '3d_axes.jpg'")