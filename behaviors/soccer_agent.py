import numpy as np

class SoccerAgent:
    """
    足球智能体战术决策层：
    负责全场态势感知、追球/绕后对齐/射门状态机切换，输出目标速度 (vx, vy, wz)。
    """
    def __init__(self, goal_pos=np.array([4.4, 0.0])):
        self.goal_pos = goal_pos
        self.state = "SEARCH"
        self.auto_mode = False

    def toggle_auto_mode(self):
        """切换自动战术模式 / 手动遥控模式"""
        self.auto_mode = not self.auto_mode
        return self.auto_mode

    def update_decision(self, robot_pos_3d, robot_quat, ball_pos_3d):
        """
        计算逼近足球与推球射门的高层速度规划
        """
        if not self.auto_mode:
            return None  # 手动模式下不接管

        robot_pos = np.array(robot_pos_3d[:2])
        ball_pos = np.array(ball_pos_3d[:2])

        # 1. 提取机器人当前 Yaw 偏航角
        w, x, y, z = robot_quat
        robot_yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # 2. 几何矢量计算
        r2b = ball_pos - robot_pos
        dist_to_ball = np.linalg.norm(r2b)
        angle_to_ball = np.arctan2(r2b[1], r2b[0])
        yaw_diff_ball = (angle_to_ball - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        # 足球指向球门向量
        b2g = self.goal_pos - ball_pos
        angle_ball_to_goal = np.arctan2(b2g[1], b2g[0])

        # 3. 规划理想接球/踢球准备点（位于足球正后方 0.35m 处）
        ideal_behind_point = ball_pos - 0.35 * np.array([np.cos(angle_ball_to_goal), np.sin(angle_ball_to_goal)])
        r2p = ideal_behind_point - robot_pos
        dist_to_behind = np.linalg.norm(r2p)
        angle_to_behind = np.arctan2(r2p[1], r2p[0])
        yaw_diff_behind = (angle_to_behind - robot_yaw + np.pi) % (2 * np.pi) - np.pi

        # 4. 状态机逻辑
        cmd_vx = 0.0
        cmd_vy = 0.0
        cmd_wz = 0.0

        if dist_to_ball > 0.45:
            # 状态 A：快速向足球后方机动
            self.state = "NAVIGATE_TO_BEHIND"
            if abs(yaw_diff_behind) > 0.4:
                cmd_wz = float(np.clip(1.5 * yaw_diff_behind, -1.2, 1.2))
                cmd_vx = 0.2
            else:
                cmd_vx = float(np.clip(0.8 * dist_to_behind, 0.3, 1.0))
                cmd_wz = float(np.clip(1.0 * yaw_diff_behind, -0.8, 0.8))
        else:
            # 状态 B：进入射门准备区，对准球门冲刺射门
            yaw_diff_goal = (angle_ball_to_goal - robot_yaw + np.pi) % (2 * np.pi) - np.pi
            if abs(yaw_diff_goal) > 0.3:
                self.state = "ALIGN_WITH_GOAL"
                cmd_wz = float(np.clip(1.8 * yaw_diff_goal, -1.2, 1.2))
                cmd_vx = 0.1
            else:
                self.state = "DRIBBLE_AND_SHOOT"
                cmd_vx = 1.0  # 冲刺推球
                cmd_wz = float(np.clip(0.8 * yaw_diff_goal, -0.5, 0.5))

        return cmd_vx, cmd_vy, cmd_wz