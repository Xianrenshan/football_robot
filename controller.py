import mujoco
import numpy as np

class Go1MotionController:
    """
    Go1 运动控制器
    支持：动态 ID 寻址、解析逆运动学 (IK)、对角小跑步态 (Trot)、跳跃动力学状态机
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.dt = model.opt.timestep

        # 1. 动态获取 12 个关节与执行器 ID (彻底消除硬编码索引)
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.actuator_names = [
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf"
        ]

        self.qpos_indices = []
        self.qvel_indices = []
        self.ctrl_indices = []

        for jname in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.qpos_indices.append(model.jnt_qposadr[jid])
                self.qvel_indices.append(model.jnt_dofadr[jid])
            else:
                raise ValueError(f"无法在模型中找到关节: {jname}")

        for aname in self.actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid != -1:
                self.ctrl_indices.append(aid)
            else:
                self.ctrl_indices = list(range(12))
                break

        # 2. Go1 连杆几何参数 (米)
        self.l_hip = 0.08      # 侧向髋关节偏移
        self.l_thigh = 0.213   # 大腿长度
        self.l_calf = 0.213    # 小腿长度

        # 4条腿髋关节根部在机身局部坐标系下的位置 [x, y]
        # 顺序: 0:FR, 1:FL, 2:RR, 3:RL
        self.leg_mount_x = [0.1881, 0.1881, -0.1881, -0.1881]
        self.leg_mount_y = [-0.04675, 0.04675, -0.04675, 0.04675]
        self.leg_side_sign = [-1.0, 1.0, -1.0, 1.0]

        # 默认静态站立足端坐标 (相对于各腿髋关节根部)
        self.nominal_foot_z = -0.30
        self.nominal_foot_y = [-0.08, 0.08, -0.08, 0.08]

        # 标准站立关节角
        self.stand_q = np.array([
            -0.05, 0.75, -1.45,  # FR
             0.05, 0.75, -1.45,  # FL
            -0.05, 0.80, -1.45,  # RR
             0.05, 0.80, -1.45   # RL
        ])

        # PD 控制器参数
        self.kp = 55.0
        self.kd = 2.8
        self.max_torque = 23.7  # 最大额定扭矩 (Nm)

        # 速度指令
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0

        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_wz = 0.0

        # Trot 步态参数
        self.gait_phase = 0.0
        self.gait_period = 0.42      # 步态周期 (s)
        self.step_height = 0.065     # 抬腿高度 (m)

        # 跳跃状态机
        self.jump_state = "IDLE"     # IDLE, CROUCH, THRUST, RETRACT, LAND
        self.jump_time = 0.0

    def set_velocity_command(self, vx, vy, wz):
        """设定目标速度指令"""
        self.target_vx = np.clip(vx, -0.8, 1.2)
        self.target_vy = np.clip(vy, -0.5, 0.5)
        self.target_wz = np.clip(wz, -1.5, 1.5)

    def trigger_jump(self):
        """触发跳跃动作"""
        if self.jump_state == "IDLE":
            self.jump_state = "CROUCH"
            self.jump_time = 0.0
            print("[动作] >>> 触发跳跃爆发！")

    def _inverse_kinematics_leg(self, x, y, z, side_sign):
        """Go1 单腿解析逆运动学 (IK)"""
        # 1. 髋侧摆角 (Roll)
        r_yz_sq = y**2 + z**2
        l_hip_sq = self.l_hip**2
        if r_yz_sq < l_hip_sq:
            r_yz_sq = l_hip_sq + 1e-5

        l_yz = np.sqrt(r_yz_sq - l_hip_sq)
        theta_base = np.arctan2(y, -z)
        theta_offset = np.arctan2(side_sign * self.l_hip, l_yz)
        q_hip = theta_base - theta_offset

        # 2. 矢状面投影长度与膝关节角 (Pitch)
        d_sq = x**2 + l_yz**2
        d = np.sqrt(d_sq)
        d = np.clip(d, 0.05, self.l_thigh + self.l_calf - 1e-4)

        cos_knee = (self.l_thigh**2 + self.l_calf**2 - d**2) / (2.0 * self.l_thigh * self.l_calf)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        q_calf = -(np.pi - np.arccos(cos_knee))

        # 3. 大腿俯仰角 (Pitch)
        beta_0 = np.arctan2(x, l_yz)
        cos_thigh = (self.l_thigh**2 + d**2 - self.l_calf**2) / (2.0 * self.l_thigh * d)
        cos_thigh = np.clip(cos_thigh, -1.0, 1.0)
        q_thigh = beta_0 + np.arccos(cos_thigh)

        return q_hip, q_thigh, q_calf

    def _update_jump_state_machine(self):
        """跳跃四阶段动力学状态机"""
        self.jump_time += self.dt
        target_q = self.stand_q.copy()
        current_kp = self.kp
        current_kd = self.kd

        if self.jump_state == "CROUCH":
            # 阶段 1: 下蹲蓄力
            crouch_z = -0.18
            for i in range(4):
                q_h, q_t, q_c = self._inverse_kinematics_leg(0.0, self.nominal_foot_y[i], crouch_z, self.leg_side_sign[i])
                target_q[i*3 : i*3+3] = [q_h, q_t, q_c]
            current_kp = 70.0
            if self.jump_time > 0.15:
                self.jump_state = "THRUST"

        elif self.jump_state == "THRUST":
            # 阶段 2: 爆发蹬地
            thrust_z = -0.38
            for i in range(4):
                q_h, q_t, q_c = self._inverse_kinematics_leg(0.0, self.nominal_foot_y[i], thrust_z, self.leg_side_sign[i])
                target_q[i*3 : i*3+3] = [q_h, q_t, q_c]
            current_kp = 95.0
            if self.jump_time > 0.27:
                self.jump_state = "RETRACT"

        elif self.jump_state == "RETRACT":
            # 阶段 3: 腾空缩腿
            retract_z = -0.22
            for i in range(4):
                q_h, q_t, q_c = self._inverse_kinematics_leg(0.0, self.nominal_foot_y[i], retract_z, self.leg_side_sign[i])
                target_q[i*3 : i*3+3] = [q_h, q_t, q_c]
            current_kp = 40.0
            if self.jump_time > 0.55:
                self.jump_state = "LAND"

        elif self.jump_state == "LAND":
            # 阶段 4: 触地缓冲阻尼
            land_z = -0.30
            for i in range(4):
                q_h, q_t, q_c = self._inverse_kinematics_leg(0.0, self.nominal_foot_y[i], land_z, self.leg_side_sign[i])
                target_q[i*3 : i*3+3] = [q_h, q_t, q_c]
            current_kp = 50.0
            current_kd = 6.5
            if self.jump_time > 0.85:
                self.jump_state = "IDLE"
                self.jump_time = 0.0

        return target_q, current_kp, current_kd

    def _generate_trot_targets(self):
        """对角小跑步态足端轨迹解算"""
        alpha = 0.08
        self.cmd_vx += alpha * (self.target_vx - self.cmd_vx)
        self.cmd_vy += alpha * (self.target_vy - self.cmd_vy)
        self.cmd_wz += alpha * (self.target_wz - self.cmd_wz)

        is_moving = (abs(self.cmd_vx) > 0.02 or abs(self.cmd_vy) > 0.02 or abs(self.cmd_wz) > 0.05)
        if is_moving:
            self.gait_phase = (self.gait_phase + self.dt / self.gait_period) % 1.0
        else:
            self.gait_phase = 0.0

        target_q = np.zeros(12)
        leg_phase_offsets = [0.0, 0.5, 0.5, 0.0]

        for i in range(4):
            phi = (self.gait_phase + leg_phase_offsets[i]) % 1.0

            vx_leg = self.cmd_vx - self.cmd_wz * self.leg_mount_y[i]
            vy_leg = self.cmd_vy + self.cmd_wz * self.leg_mount_x[i]

            step_len_x = vx_leg * (self.gait_period * 0.5)
            step_len_y = vy_leg * (self.gait_period * 0.5)

            if not is_moving:
                foot_x = 0.0
                foot_y = self.nominal_foot_y[i]
                foot_z = self.nominal_foot_z
            elif phi < 0.5:
                # 摆动相
                tau = phi / 0.5
                foot_x = -step_len_x * np.cos(np.pi * tau)
                foot_y = self.nominal_foot_y[i] - step_len_y * np.cos(np.pi * tau)
                foot_z = self.nominal_foot_z + self.step_height * np.sin(np.pi * tau)
            else:
                # 支撑相
                tau = (phi - 0.5) / 0.5
                foot_x = step_len_x * (1.0 - 2.0 * tau)
                foot_y = self.nominal_foot_y[i] + step_len_y * (1.0 - 2.0 * tau)
                foot_z = self.nominal_foot_z

            q_h, q_t, q_c = self._inverse_kinematics_leg(foot_x, foot_y, foot_z, self.leg_side_sign[i])
            target_q[i*3 : i*3+3] = [q_h, q_t, q_c]

        return target_q, self.kp, self.kd

    def update(self):
        """控制器主循环：计算并输出 12 个关节电机力矩"""
        if self.jump_state != "IDLE":
            target_q, kp, kd = self._update_jump_state_machine()
        else:
            target_q, kp, kd = self._generate_trot_targets()

        # 动态读取 12 关节位置与角速度
        current_q = self.data.qpos[self.qpos_indices]
        current_dq = self.data.qvel[self.qvel_indices]

        # 计算 PD 扭矩
        torques = kp * (target_q - current_q) - kd * current_dq

        # 输出到执行器
        clipped_torques = np.clip(torques, -self.max_torque, self.max_torque)
        for i, aid in enumerate(self.ctrl_indices):
            self.data.ctrl[aid] = clipped_torques[i]