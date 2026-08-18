import os
import io
import pickle
import mujoco
import numpy as np
import torch
from action_config import (
    BODY_MODEL_PATH,
    ADAPTATION_MODEL_PATH,
    PARAMS_PATH,
    POLICY_DT,
    MAX_VX,
    MIN_VX,
    MAX_VY,
    MAX_WZ,
    DEFAULT_BODY_HEIGHT,
    DEFAULT_GAIT_FREQ,
    DEFAULT_GAIT_PHASE,
    DEFAULT_GAIT_OFFSET,
    DEFAULT_GAIT_BOUND,
    DEFAULT_GAIT_DURATION,
    DEFAULT_FOOTSWING_HEIGHT,
    DEFAULT_BODY_PITCH,
    DEFAULT_BODY_ROLL,
    DEFAULT_STANCE_WIDTH,
    DEFAULT_STANCE_LENGTH,
    DEFAULT_AUX_REWARD,
    DEFAULT_KP,
    DEFAULT_KD,
    ACTION_SCALE,
    MAX_TORQUE
)


class CPU_Unpickler(pickle.Unpickler):
    """自定义反序列化器：将 pickle 中保存的 CUDA 存储对象安全重定向到 CPU 内存"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


class Go1MotionController:
    """
    基于 MIT Walk-These-Ways 预训练大脑 (TorchScript) 的 Go1 运动控制器。
    解决 CPU 跨设备反序列化问题，严格对齐 70 维单步观测与 2100 维时序历史输入，
    实现原地不动如山的自平衡站立与全向连续高动态运控。
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.sim_dt = model.opt.timestep

        # 1. 策略标准的 12 关节命名与顺序 (FL, FR, RL, RR)
        self.policy_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]

        # 策略对应的执行器名称
        self.policy_actuator_names = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf"
        ]

        # 建立 MuJoCo 索引映射
        self.qpos_indices = []
        self.qvel_indices = []
        self.ctrl_indices = []

        for jname in self.policy_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1:
                self.qpos_indices.append(model.jnt_qposadr[jid])
                self.qvel_indices.append(model.jnt_dofadr[jid])
            else:
                raise ValueError(f"无法在模型中找到关节: {jname}")

        for aname in self.policy_actuator_names:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid != -1:
                self.ctrl_indices.append(aid)
            else:
                self.ctrl_indices = list(range(12))
                break

        self.trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

        # 2. 加载模型参数与 TorchScript 神经网络
        self._load_policy_and_parameters()

        # 3. 控制分频与时钟管理
        self.control_dt = POLICY_DT
        self.decimation = max(1, int(round(self.control_dt / self.sim_dt)))
        self.step_counter = 0

        # 4. 指令与步态时钟状态
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.gait_indices = torch.zeros(1, dtype=torch.float32)

        self.last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.last_last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.current_target_dof_pos = self.default_dof_pos.numpy().copy()

        # 5. 初始化 70 维特征与 30 步历史队列 (总长度 2100 维)
        self.history_len = getattr(self, "num_obs_history", 30)
        self.num_obs = getattr(self, "num_observations", 70)
        self.obs_history_list = []
        self.reset_state()

    def _load_policy_and_parameters(self):
        """安全读取 parameters.pkl 并通过 TorchScript 加载策略网络"""
        if not os.path.exists(BODY_MODEL_PATH) or not os.path.exists(ADAPTATION_MODEL_PATH):
            raise FileNotFoundError("未在 assets 目录找到 body_latest.jit 或 adaptation_module_latest.jit！")

        print("[策略] 正在加载 TorchScript 模型与参数...")
        self.body_model = torch.jit.load(BODY_MODEL_PATH, map_location="cpu")
        self.adaptation_module = torch.jit.load(ADAPTATION_MODEL_PATH, map_location="cpu")
        self.body_model.eval()
        self.adaptation_module.eval()

        # 默认出厂参数
        default_angles_dict = {
            "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        }
        self.kp = DEFAULT_KP
        self.kd = DEFAULT_KD
        self.action_scale = ACTION_SCALE
        self.num_obs_history = 30
        self.num_observations = 70

        # 观测与指令缩放系数完整默认声明 (补齐 gait_offset / gait_bound / gait_duration)
        self.scale_lin_vel = 2.0
        self.scale_ang_vel = 0.25
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_body_height = 2.0
        self.scale_gait_freq = 1.0
        self.scale_gait_phase = 1.0
        self.scale_gait_offset = 1.0
        self.scale_gait_bound = 1.0
        self.scale_gait_duration = 1.0
        self.scale_footswing_height = 0.15
        self.scale_body_pitch = 0.3
        self.scale_body_roll = 0.3
        self.scale_stance_width = 1.0
        self.scale_stance_length = 1.0
        self.scale_aux_reward = 1.0

        # 使用 CPU_Unpickler 安全读取 parameters.pkl
        if os.path.exists(PARAMS_PATH):
            try:
                try:
                    params = torch.load(PARAMS_PATH, map_location="cpu")
                except Exception:
                    with open(PARAMS_PATH, "rb") as f:
                        params = CPU_Unpickler(f).load()

                cfg = params.get("Cfg", {})
                if isinstance(cfg, dict):
                    # 提取关节默认角
                    init_state = cfg.get("init_state", {})
                    if "default_joint_angles" in init_state:
                        default_angles_dict = init_state["default_joint_angles"]

                    # 提取 PD 增益与动作尺度
                    control = cfg.get("control", {})
                    stiffness = control.get("stiffness", {})
                    damping = control.get("damping", {})
                    if stiffness:
                        self.kp = float(list(stiffness.values())[0])
                    if damping:
                        self.kd = float(list(damping.values())[0])
                    self.action_scale = float(control.get("action_scale", ACTION_SCALE))

                    # 提取历史窗口与观测维度
                    env_cfg = cfg.get("env", {})
                    self.num_obs_history = int(env_cfg.get("num_observation_history", 30))
                    self.num_observations = int(env_cfg.get("num_observations", 70))

                    obs_scales = cfg.get("normalization", {}).get("obs_scales", {})
                    if obs_scales:
                        self.scale_lin_vel = float(obs_scales.get("lin_vel", self.scale_lin_vel))
                        self.scale_ang_vel = float(obs_scales.get("ang_vel", self.scale_ang_vel))
                        self.scale_dof_pos = float(obs_scales.get("dof_pos", self.scale_dof_pos))
                        self.scale_dof_vel = float(obs_scales.get("dof_vel", self.scale_dof_vel))
                        self.scale_body_height = float(obs_scales.get("body_height", self.scale_body_height))
                        self.scale_gait_freq = float(obs_scales.get("gait_freq", self.scale_gait_freq))
                        self.scale_gait_phase = float(obs_scales.get("gait_phase", self.scale_gait_phase))
                        self.scale_gait_offset = float(obs_scales.get("gait_offset", self.scale_gait_offset))
                        self.scale_gait_bound = float(obs_scales.get("gait_bound", self.scale_gait_bound))
                        self.scale_gait_duration = float(obs_scales.get("gait_duration", self.scale_gait_duration))
                        self.scale_footswing_height = float(obs_scales.get("footswing_height", self.scale_footswing_height))
                        self.scale_body_pitch = float(obs_scales.get("body_pitch", self.scale_body_pitch))
                        self.scale_body_roll = float(obs_scales.get("body_roll", self.scale_body_roll))
                        self.scale_stance_width = float(obs_scales.get("stance_width", self.scale_stance_width))
                        self.scale_stance_length = float(obs_scales.get("stance_length", self.scale_stance_length))
                        self.scale_aux_reward = float(obs_scales.get("aux_reward", self.scale_aux_reward))
                print("[策略] 成功从 parameters.pkl 读取真实训练配置！")
            except Exception as e:
                print(f"[警告] 读取 parameters.pkl 发生异常，使用标准默认配置: {e}")

        # 整理默认关节角度张量
        default_pos_list = [default_angles_dict.get(name, 0.0) for name in self.policy_joint_names]
        self.default_dof_pos = torch.tensor(default_pos_list, dtype=torch.float32)

        # 构造 15 维指令尺度向量
        self.commands_scale = torch.tensor([
            self.scale_lin_vel, self.scale_lin_vel, self.scale_ang_vel,
            self.scale_body_height, self.scale_gait_freq,
            self.scale_gait_phase, self.scale_gait_offset, self.scale_gait_bound, self.scale_gait_duration,
            self.scale_footswing_height, self.scale_body_pitch, self.scale_body_roll,
            self.scale_stance_width, self.scale_stance_length, self.scale_aux_reward
        ], dtype=torch.float32)

        print(f"[策略] 初始化就绪: Kp={self.kp}, Kd={self.kd}, 单帧维度={self.num_observations}, 历史窗口={self.num_obs_history} (总特征={self.num_obs_history*self.num_observations})")

    def set_velocity_command(self, vx, vy, wz):
        """设定目标速度向量 (vx, vy, wz)"""
        self.cmd_vx = np.clip(vx, MIN_VX, MAX_VX)
        self.cmd_vy = np.clip(vy, -MAX_VY, MAX_VY)
        self.cmd_wz = np.clip(wz, -MAX_WZ, MAX_WZ)

    def _get_projected_gravity(self):
        """计算机身坐标系下的重力投影向量 (3维)"""
        w, x, y, z = self.data.xquat[self.trunk_body_id]
        gx = 2.0 * (x * z - w * y)
        gy = 2.0 * (y * z + w * x)
        gz = -(1.0 - 2.0 * (x * x + y * y))
        return torch.tensor([gx, gy, gz], dtype=torch.float32)

    def _get_current_single_observation(self):
        """
        组装符合 Walk-These-Ways 规范的标准 70 维单步观测特征：
        - 机身角速度 (3维)
        - 重力投影向量 (3维)
        - 高层运动指令向量 (15维)
        - 关节位置偏差 (12维)
        - 关节角速度 (12维)
        - 当前动作 (12维)
        - 上上步历史动作 (12维)
        - 步态时钟信号 (1维)
        """
        # 1. 重力投影 (3维) 与 机身角速度 (3维)
        projected_gravity = self._get_projected_gravity().unsqueeze(0)
        cvel = self.data.cvel[self.trunk_body_id]
        base_ang_vel = torch.tensor(cvel[0:3], dtype=torch.float32).unsqueeze(0) * self.scale_ang_vel

        # 2. 指令向量 (15维)
        commands = torch.tensor([
            self.cmd_vx, self.cmd_vy, self.cmd_wz,
            DEFAULT_BODY_HEIGHT, DEFAULT_GAIT_FREQ,
            DEFAULT_GAIT_PHASE, DEFAULT_GAIT_OFFSET, DEFAULT_GAIT_BOUND, DEFAULT_GAIT_DURATION,
            DEFAULT_FOOTSWING_HEIGHT, DEFAULT_BODY_PITCH, DEFAULT_BODY_ROLL,
            DEFAULT_STANCE_WIDTH, DEFAULT_STANCE_LENGTH, DEFAULT_AUX_REWARD
        ], dtype=torch.float32)
        scaled_commands = (commands * self.commands_scale).unsqueeze(0)

        # 3. 关节位置偏差 (12维) 与 速度 (12维)
        current_q = self.data.qpos[self.qpos_indices]
        current_dq = self.data.qvel[self.qvel_indices]

        dof_pos_tensor = torch.tensor(current_q, dtype=torch.float32)
        dof_vel_tensor = torch.tensor(current_dq, dtype=torch.float32)

        dof_pos_scaled = ((dof_pos_tensor - self.default_dof_pos) * self.scale_dof_pos).unsqueeze(0)
        dof_vel_scaled = (dof_vel_tensor * self.scale_dof_vel).unsqueeze(0)

        # 4. 动作历史 (12维 + 12维)
        actions = self.last_action
        last_actions = self.last_last_action

        # 5. 步态时钟相位输入 (1维)
        timing_inputs = self.gait_indices.unsqueeze(0)

        # 6. 拼接构成 70 维特征: 3 + 3 + 15 + 12 + 12 + 12 + 12 + 1 = 70
        single_obs = torch.cat([
            projected_gravity,
            base_ang_vel,
            scaled_commands,
            dof_pos_scaled,
            dof_vel_scaled,
            actions,
            last_actions,
            timing_inputs
        ], dim=-1)

        # 严格校验维度：若与模型声明维度不一致则进行稳健对齐
        if single_obs.shape[-1] < self.num_observations:
            pad = torch.zeros((1, self.num_observations - single_obs.shape[-1]), dtype=torch.float32)
            single_obs = torch.cat([single_obs, pad], dim=-1)
        elif single_obs.shape[-1] > self.num_observations:
            single_obs = single_obs[:, :self.num_observations]

        return single_obs

    def _run_policy_inference(self):
        """执行策略网络前向推理 (50Hz)"""
        # 1. 步态时钟相位推进
        is_moving = (abs(self.cmd_vx) > 0.01 or abs(self.cmd_vy) > 0.01 or abs(self.cmd_wz) > 0.02)
        if is_moving:
            self.gait_indices = (self.gait_indices + DEFAULT_GAIT_FREQ * self.control_dt) % 1.0
        else:
            self.gait_indices.zero_()

        # 2. 提取 70 维单帧观测并更新历史队列
        single_obs = self._get_current_single_observation()
        self.obs_history_list.append(single_obs)
        if len(self.obs_history_list) > self.history_len:
            self.obs_history_list.pop(0)

        # 3. 拼接生成严格符合 (1, 2100) 尺寸的时序张量
        obs_history = torch.cat(self.obs_history_list, dim=-1)

        # 4. 执行双脑协同前向推理
        with torch.no_grad():
            latent = self.adaptation_module(obs_history)
            policy_input = torch.cat([obs_history, latent], dim=-1)
            actions = self.body_model(policy_input)

        # 5. 更新动作历史与目标关节角度
        self.last_last_action = self.last_action.clone()
        self.last_action = actions.clone()

        action_np = actions.squeeze(0).cpu().numpy()
        self.current_target_dof_pos = self.default_dof_pos.numpy() + self.action_scale * action_np

    def update(self):
        """控制器主循环：分频调用策略推理，高频下发关节 PD 力矩"""
        # 1. 达到分频周期时触发 50Hz 策略前向推理
        if self.step_counter % self.decimation == 0:
            self._run_policy_inference()
        self.step_counter += 1

        # 2. 500Hz 高频关节 PD 伺服
        current_q = self.data.qpos[self.qpos_indices]
        current_dq = self.data.qvel[self.qvel_indices]

        torques = self.kp * (self.current_target_dof_pos - current_q) - self.kd * current_dq
        torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

        # 写入 MuJoCo 执行器
        for i, aid in enumerate(self.ctrl_indices):
            self.data.ctrl[aid] = torques[i]

    def reset_state(self):
        """重置控制器的所有内部状态"""
        self.step_counter = 0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.gait_indices.zero_()
        self.last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.last_last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.current_target_dof_pos = self.default_dof_pos.numpy().copy()

        # 用初始静态观测完整灌满 30 步历史窗口 (生成完整的 2100 维初始数据)
        self.obs_history_list = []
        init_single_obs = self._get_current_single_observation()
        for _ in range(self.history_len):
            self.obs_history_list.append(init_single_obs.clone())