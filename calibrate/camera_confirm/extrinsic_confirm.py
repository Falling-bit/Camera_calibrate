#   based on intrinsic_confirm.py with intrinsic already confirmed
#   and use this extrinsic_confirm.py to get remained extrinsic confirm
#   output errors and all_coeffs

import cv2
import numpy as np
import glob
from datetime import datetime
import pandas as pd

# 棋盘格参数
from chessboard import x_num, y_num
from intrinsic_confirm import camera_matrix,dist_coeffs

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
image_names = []


rvecs = []
tvecs = []

# 准备Excel输出数据结构
results = {
    'Image Name': [],
    'Reprojection Error': [],
    'Camera Matrix': [],
    'Rotation Vector (rvec)': [],
    'Translation Vector (tvec)': [],
    'Distortion Coefficients': []
}



# 检测角点
images = glob.glob(r'./r&tvec/*.jpg')   # 获取所有jpg和png图片
print(images)
for image in images:
    img = cv2.imread(image)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

    if ret:
        # 亚像素优化
        corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        world_points.append(world)
        image_points.append(corners_subpix)
        image_names.append(image)

        # 可视化角点
        cv2.drawChessboardCorners(img, (num_x, num_y), corners_subpix, ret)
        cv2.imwrite(f'corners_{image}', img)


if not world_points:
    print("No chessboard found in any images!")
    exit()

# 相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        world_points, image_points, gray.shape[::-1], camera_matrix, dist_coeffs
)
print(camera_matrix)

#calcu mean
avr_rvecs = np.mean(rvecs,axis=0)
avr_tvecs = np.mean(tvecs,axis=0)

# 计算重投影误差并收集结果
for i in range(len(world_points)):  #world points 份数
    img_points_reproj, _ = cv2.projectPoints(world_points[i], avr_rvecs, avr_tvecs, camera_matrix, dist_coeffs)
    error = cv2.norm(image_points[i], img_points_reproj, cv2.NORM_L2) / len(img_points_reproj)

    '''
    # 输出旋转角（欧拉角）
    rmat, _ = cv2.Rodrigues(rvecs[i])
    euler_angles = np.degrees(rotationMatrixToEulerAngles(rmat))  # 转为角度
    print(f"Euler angles (deg): Roll={euler_angles[0]:.1f}, Pitch={euler_angles[1]:.1f}, Yaw={euler_angles[2]:.1f}")
    '''

    # 保存每张图片的结果
    results['Image Name'].append(image_names[i])
    results['Reprojection Error'].append(error)
    results['Rotation Vector (rvec)'].append(rvecs[i].flatten())
    results['Translation Vector (tvec)'].append(tvecs[i].flatten())
    results['Camera Matrix'].append(camera_matrix.flatten())
    results['Distortion Coefficients'].append(dist_coeffs.flatten())




# 创建DataFrame并保存到Excel
df = pd.DataFrame(results)
excel_filename = f'calibration_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
df.to_excel(excel_filename, index=False)

# 保存全局标定参数
with open('global_calibration_results.txt', 'w') as f:
    f.write(f"=== Global Camera Calibration Results (Generated on {datetime.now()}) ===\n\n")
    f.write(f"Chessboard Square Size: {square_size} mm\n")
    f.write(f"Chessboard Dimensions: {num_x} x {num_y} (inner corners)\n")
    f.write(f"Mean Reprojection Error: {np.mean(results['Reprojection Error']):.4f} pixels\n\n")

    f.write("Rotation Vector:\n")
    np.savetxt(f,avr_rvecs,fmt='%.8f')

    f.write("Translation Vector:\n")
    np.savetxt(f,avr_tvecs,fmt='%.8f')

    f.write("\nCamera Matrix (Intrinsic):\n")
    np.savetxt(f, camera_matrix, fmt='%.8f')

    f.write("\nDistortion Coefficients (k1, k2, p1, p2, k3):\n")
    np.savetxt(f, dist_coeffs.reshape(1, -1), fmt='%.8f')

print(f"Calibration completed! Individual results saved to {excel_filename}")
print(f"Global calibration parameters saved to 'global_calibration_results.txt'")