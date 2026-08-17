import numpy as np

class ThirdPersonCameraController:
    """
    第三人称相机控制器（轻量版）
    只负责：
      - 从四元数提取偏航角
      - 对偏航角和俯仰角进行独立低通滤波（解决闪烁）
      - 计算注视点（躯干位置 + 垂直偏移）
    主循环直接使用滤波后的角度和注视点，不额外计算位置。
    """
    def __init__(self,
                 distance=1.8,
                 elevation_deg=-20.0,
                 lookat_height_offset=0.35,
                 smoothness_azimuth=0.10,
                 smoothness_elevation=0.10):
        """
        distance          : 固定相机距离（主循环设置）
        elevation_deg     : 目标俯仰角（度，负值表示俯视）
        lookat_height_offset : 注视点高于躯干的高度偏移
        smoothness_azimuth   : 偏航角滤波系数 (0~1)，越小越平滑
        smoothness_elevation : 俯仰角滤波系数
        """
        self.distance = distance
        self.target_elevation = np.radians(elevation_deg)
        self.lookat_offset = lookat_height_offset
        self.alpha_az = smoothness_azimuth
        self.alpha_el = smoothness_elevation

        # 滤波缓存（None 表示未初始化）
        self.filtered_azimuth = None
        self.filtered_elevation = None

    def reset(self):
        """重置滤波状态（场景重置时调用）"""
        self.filtered_azimuth = None
        self.filtered_elevation = None

    def update(self, robot_pos, robot_quat):
        """
        输入：
          robot_pos  : 躯干位置 (x, y, z)  (numpy array 或 list)
          robot_quat : 躯干四元数 (w, x, y, z) (MuJoCo 顺序)
        返回：
          azimuth_deg : 滤波后的方位角（度）
          elevation_deg : 滤波后的俯仰角（度）
          lookat      : 注视点 (x, y, z) numpy array
        """
        # 1. 从四元数提取偏航角（Yaw）
        w, x, y, z = robot_quat
        yaw = np.arctan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

        # 目标方位角：始终位于机器人正后方（yaw + π）
        target_az = yaw + np.pi
        # 目标俯仰角：固定
        target_el = self.target_elevation

        # 2. 低通滤波（带角度环绕处理）
        if self.filtered_azimuth is None:
            self.filtered_azimuth = target_az
            self.filtered_elevation = target_el
        else:
            # 方位角差值，约束到 [-π, π]
            delta_az = target_az - self.filtered_azimuth
            delta_az = (delta_az + np.pi) % (2 * np.pi) - np.pi
            self.filtered_azimuth += self.alpha_az * delta_az

            # 俯仰角差值（无环绕问题）
            delta_el = target_el - self.filtered_elevation
            self.filtered_elevation += self.alpha_el * delta_el

        # 3. 注视点：躯干位置 + 垂直偏移
        lookat = np.array(robot_pos, dtype=float)
        lookat[2] += self.lookat_offset

        # 返回角度（度）
        return np.degrees(self.filtered_azimuth), np.degrees(self.filtered_elevation), lookat