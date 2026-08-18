import numpy as np

class ThirdPersonCameraController:
    """
    轻量级第三人称低通滤波相机控制器：
    跟踪机器人躯干位置与朝向，消除高频视角晃动与反射光斑频闪。
    """
    def __init__(self,
                 distance=1.8,
                 elevation_deg=-20.0,
                 lookat_height_offset=0.30,
                 smoothness_azimuth=0.12,
                 smoothness_elevation=0.12):
        self.distance = distance
        self.target_elevation = np.radians(elevation_deg)
        self.lookat_offset = lookat_height_offset
        self.alpha_az = smoothness_azimuth
        self.alpha_el = smoothness_elevation

        self.filtered_azimuth = None
        self.filtered_elevation = None

    def reset(self):
        """重置滤波缓存"""
        self.filtered_azimuth = None
        self.filtered_elevation = None

    def update(self, robot_pos, robot_quat):
        """根据机器人位置和四元数更新注视点与方位角"""
        w, x, y, z = robot_quat
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        target_az = yaw + np.pi
        target_el = self.target_elevation

        if self.filtered_azimuth is None:
            self.filtered_azimuth = target_az
            self.filtered_elevation = target_el
        else:
            delta_az = (target_az - self.filtered_azimuth + np.pi) % (2 * np.pi) - np.pi
            self.filtered_azimuth += self.alpha_az * delta_az

            delta_el = target_el - self.filtered_elevation
            self.filtered_elevation += self.alpha_el * delta_el

        lookat = np.array(robot_pos, dtype=float)
        lookat[2] += self.lookat_offset

        return np.degrees(self.filtered_azimuth), np.degrees(self.filtered_elevation), lookat