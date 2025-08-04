import cv2
import numpy as np
from _2dto3d import s,camera_matrix,rvecs,tvecs,dist_coeffs



class Interactive3DViewer:
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvec = None
        self.tvec = None
        self.current_img = None
        self.window_name = "3D Coordinate Viewer"

    def load(self, img_path):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvec = rvecs
        self.tvec = tvecs
        self.current_img = cv2.imread(img_path)
        self.window_name = "3D Coordinate Viewer"


    def unproject_points(self, pixel_points, Z=1.0):
        # 正确获取单个图像点坐标
        u, v = pixel_points[0][0], pixel_points[0][1]  # 获取(u,v)坐标
        print('Input image point:', (u, v))

        # 将图像点转换为齐次坐标 (3x1)
        image_point = np.array([[u], [v], [Z]], dtype=np.float32)

        # 计算相机坐标系下的点
        cam_point = np.linalg.inv(camera_matrix) @ image_point * s

        # 转换到世界坐标系
        R, _ = cv2.Rodrigues(rvecs)
        XY_world = np.linalg.inv(R) @ (cam_point - tvecs.reshape(3, 1))

        return XY_world

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_MOUSEMOVE:
            # 反投影到3D坐标(Z=0平面)
            pixel_point = np.array([[x, y]])
            world_point = self.unproject_points(pixel_point, Z=1.0)

            X, Y = world_point[0], world_point[1]
            print(f'output world: ({X[0]},{Y[0]})')           # 显示坐标
            display_img = self.current_img.copy()
            cv2.putText(display_img,
                        f"World Coord: X={X[0]:.1f}mm, Y={Y[0]:.1f}mm, Z=1.0mm",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.imshow(self.window_name, display_img)

    def run(self,img_path):
        """运行交互式查看器"""
        self.load(img_path)
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        cv2.imshow(self.window_name, self.current_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = r'./r&tvec/20250728160418_43.jpg'
    #Interactive3DViewer.__init__(self=Interactive3DViewer,img_path=img_path)
    viewer = Interactive3DViewer()
    viewer.run(img_path=img_path)