import os
import io
import pickle
import mujoco
import numpy as np
import torch
from config.action_config import (
    BODY_MODEL_PATH,
    ADAPTATION_MODEL_PATH,
    POLICY_DT,
    SIM_DT,
    DECIMATION,
    MAX_VX,
    MIN_VX,
    MAX_VY,
    MAX_WZ,
    DEFAULT_KP,
    DEFAULT_KD,
    ACTION_SCALE,
    MAX_TORQUE,
    DEFAULT_GAIT_FREQ
)
from core.sensor_adapter import SensorAdapter
from core.observation_manager import ObservationManager

class Go1MotionController:
    """
    Go1 运动控制器：严格按照 Walk-These-Ways 50Hz 策略前向推理与 500Hz PD 闭环伺服
    """
    def __init__(self, model, data, hip_scale_reduction=0.5):
        self.model = model
        self.data = data
        self.sim_dt = SIM_DT
        self.control_dt = POLICY_DT
        self.decimation = DECIMATION
        self.hip_scale_reduction = hip_scale_reduction

        # 1. 关节拓扑 (FL, FR, RL, RR)
        self.policy_joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
        ]

        self.qpos_indices = []
        self.qvel_indices = []
        self.ctrl_indices = []

        for jname in self.policy_joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                raise ValueError(f"模型中找不到关节: {jname}")
            self.qpos_indices.append(model.jnt_qposadr[jid])
            self.qvel_indices.append(model.jnt_dofadr[jid])

        for aname in ["FL_hip", "FL_thigh", "FL_calf", "FR_hip", "FR_thigh", "FR_calf",
                      "RL_hip", "RL_thigh", "RL_calf", "RR_hip", "RR_thigh", "RR_calf"]:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid != -1:
                self.ctrl_indices.append(aid)
            else:
                self.ctrl_indices = list(range(12))
                break

        # 2. 标称关节角度 (与 WTW go1_config 一致)
        self.default_angles_dict = {
            "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
        }
        default_pos = [self.default_angles_dict[j] for j in self.policy_joint_names]
        self.default_dof_pos = torch.tensor(default_pos, dtype=torch.float32)

        # 3. 运控超参数
        self.kp = DEFAULT_KP
        self.kd = DEFAULT_KD
        self.action_scale = ACTION_SCALE

        # 4. 初始化
        self.sensor_adapter = SensorAdapter(model, data, trunk_body_name="trunk")
        self.obs_manager = ObservationManager(self.default_dof_pos)

        # 5. 加载 TorchScript 模型
        self._load_policy_models()

        # 6. 控制时钟与内部状态
        self.step_counter = 0
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.gait_indices = 0.0

        self.last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.last_last_action = torch.zeros((1, 12), dtype=torch.float32)
        self.current_target_dof_pos = self.default_dof_pos.numpy().copy()

    def _load_policy_models(self):
        if not os.path.exists(BODY_MODEL_PATH) or not os.path.exists(ADAPTATION_MODEL_PATH):
            raise FileNotFoundError("未在 assets 目录找到策略权重文件！")

        print("[运控层] 正在加载 TorchScript 模型...")
        self.body_model = torch.jit.load(BODY_MODEL_PATH, map_location="cpu")
        self.adaptation_module = torch.jit.load(ADAPTATION_MODEL_PATH, map_location="cpu")
        self.body_model.eval()
        self.adaptation_module.eval()

    def set_velocity_command(self, vx, vy, wz):
        self.cmd_vx = float(np.clip(vx, MIN_VX, MAX_VX))
        self.cmd_vy = float(np.clip(vy, -MAX_VY, MAX_VY))
        self.cmd_wz = float(np.clip(wz, -MAX_WZ, MAX_WZ))

    def _run_policy_inference(self):
        """50Hz 策略前向推理"""
        # 1. 步态时钟累加: gait_indices = (gait_indices + dt * freq) % 1.0
        freq = float(self.obs_manager.default_mob_params[1])
        self.gait_indices = (self.gait_indices + self.control_dt * freq) % 1.0

        # 2. 提取传感器物理量
        proj_gravity = self.sensor_adapter.get_projected_gravity()
        dof_pos, dof_vel = self.sensor_adapter.get_joint_states(self.qpos_indices, self.qvel_indices)

        # 3. 装配标准 70 维特征并压入历史序列
        single_obs = self.obs_manager.build_single_observation(
            proj_gravity, (self.cmd_vx, self.cmd_vy, self.cmd_wz),
            dof_pos, dof_vel, self.last_action, self.last_last_action, self.gait_indices
        )
        obs_history = self.obs_manager.push_and_get_history(single_obs)

        # 4. 双脑协同推理: [obs_history(2100), latent(2)] -> 2102维输入
        with torch.no_grad():
            latent = self.adaptation_module(obs_history)
            policy_input = torch.cat([obs_history, latent], dim=-1)
            actions = self.body_model(policy_input)

        # 5. 更新历史动作
        self.last_last_action = self.last_action.clone()
        self.last_action = actions.clone()

        # 6. 计算目标关节角度 (应用 hip_scale_reduction)
        action_np = actions.squeeze(0).cpu().numpy().copy()
        action_np[[0, 3, 6, 9]] *= self.hip_scale_reduction

        self.current_target_dof_pos = self.default_dof_pos.numpy() + self.action_scale * action_np

    def update(self):
        """500Hz 物理更新 + 50Hz 策略推理"""
        if self.step_counter % self.decimation == 0:
            self._run_policy_inference()

        # 500Hz PD 阻抗力矩控制
        current_q = self.data.qpos[self.qpos_indices]
        current_dq = self.data.qvel[self.qvel_indices]
        torques = self.kp * (self.current_target_dof_pos - current_q) - self.kd * current_dq
        torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

        for i, aid in enumerate(self.ctrl_indices):
            self.data.ctrl[aid] = torques[i]

        self.step_counter += 1

    def reset_state(self):
        self.step_counter = 0
        self.gait_indices = 0.0
        self.last_action.zero_()
        self.last_last_action.zero_()
        self.current_target_dof_pos = self.default_dof_pos.numpy().copy()

        proj_gravity = self.sensor_adapter.get_projected_gravity()
        dof_pos, dof_vel = self.sensor_adapter.get_joint_states(self.qpos_indices, self.qvel_indices)

        init_obs = self.obs_manager.build_single_observation(
            proj_gravity, (0.0, 0.0, 0.0),
            dof_pos, dof_vel, self.last_action, self.last_last_action, 0.0
        )
        self.obs_manager.reset_history(init_obs)