import os
import json
import torch
import numpy as np
from config.action_config import (
    POLICY_CONFIG_PATH,
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
    DEFAULT_AUX_REWARD
)

class ObservationManager:
    """
    严格按照 Walk-These-Ways 官方 70 维特征通道与 4 维步态时钟装配特征。
    """
    def __init__(self, default_dof_pos_tensor, clip_observations=100.0):
        self.default_dof_pos = default_dof_pos_tensor
        self.num_obs = 70
        self.num_obs_history = 30
        self.total_history_dim = self.num_obs * self.num_obs_history
        self.clip_observations = clip_observations

        # 官方归一化尺度
        self.scales = {
            "lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05,
            "body_height_cmd": 2.0, "gait_freq_cmd": 1.0, "gait_phase_cmd": 1.0,
            "footswing_height_cmd": 0.15, "body_pitch_cmd": 0.3, "body_roll_cmd": 0.3,
            "stance_width_cmd": 1.0, "stance_length_cmd": 1.0, "aux_reward_cmd": 1.0
        }
        self._build_command_scale_tensor()

        # 默认 12 维 MoB 步态参数 (对角 Trot 步态)
        self.default_mob_params = torch.tensor([
            0.0,   # body_height_cmd
            3.0,   # gait_freq_cmd (3.0 Hz)
            0.5,   # gait_phase_cmd (0.5 为 Trot)
            0.0,   # gait_offset_cmd
            0.0,   # gait_bound_cmd
            0.5,   # gait_duration_cmd
            0.08,  # footswing_height_cmd
            0.0,   # body_pitch_cmd
            0.0,   # body_roll_cmd
            0.30,  # stance_width_cmd
            0.40,  # stance_length_cmd
            0.0    # aux_reward_cmd
        ], dtype=torch.float32)

        self._obs_history_flat = None

    def _build_command_scale_tensor(self):
        """构建官方标准的 15 维指令缩放系数向量"""
        self.commands_scale = torch.tensor([
            self.scales["lin_vel"],              # vx
            self.scales["lin_vel"],              # vy
            self.scales["ang_vel"],              # wz
            self.scales["body_height_cmd"],      # body_height
            self.scales["gait_freq_cmd"],        # gait_freq
            self.scales["gait_phase_cmd"],       # gait_phase
            self.scales["gait_phase_cmd"],       # gait_offset
            self.scales["gait_phase_cmd"],       # gait_bound
            self.scales["gait_phase_cmd"],       # gait_duration
            self.scales["footswing_height_cmd"], # footswing_height
            self.scales["body_pitch_cmd"],       # body_pitch
            self.scales["body_roll_cmd"],        # body_roll
            self.scales["stance_width_cmd"],     # stance_width
            self.scales["stance_length_cmd"],    # stance_length
            self.scales["aux_reward_cmd"]        # aux_reward
        ], dtype=torch.float32).unsqueeze(0)

    def compute_clock_inputs(self, gait_indices, phase=0.5, offset=0.0, bound=0.0):
        """
        计算四足 4 维时钟正弦输入 (FL, FR, RL, RR)
        """
        foot_indices = torch.tensor([
            gait_indices + phase + offset + bound, # FL
            gait_indices + offset,                 # FR
            gait_indices + bound,                  # RL
            gait_indices + phase                   # RR
        ], dtype=torch.float32)
        clock_inputs = torch.sin(2.0 * np.pi * foot_indices).unsqueeze(0) # (1, 4)
        return clock_inputs

    def build_single_observation(self, proj_gravity, raw_vel_cmd, dof_pos, dof_vel, 
                                 last_action, last_last_action, gait_indices):
        """
        严格匹配 WTW 官方 70 维结构：
        [0:3]   projected_gravity (3维)
        [3:18]  commands * commands_scale (15维)
        [18:30] dof_pos_residual * scale (12维)
        [30:42] dof_vel * scale (12维)
        [42:54] actions (上一周期网络输出, 12维)
        [54:66] last_actions (上上周期历史动作, 12维)
        [66:70] clock_inputs (4维四足时钟输入)
        """
        # 1. 构造 15 维指令
        vel_cmd_t = torch.tensor(raw_vel_cmd, dtype=torch.float32)
        raw_cmd = torch.cat([vel_cmd_t, self.default_mob_params]).unsqueeze(0)
        scaled_commands = raw_cmd * self.commands_scale # (1, 15)

        # 2. 构造关节特征
        proj_grav_obs = proj_gravity.unsqueeze(0) if proj_gravity.ndim == 1 else proj_gravity
        dof_pos_obs = ((dof_pos - self.default_dof_pos) * self.scales["dof_pos"]).unsqueeze(0) if dof_pos.ndim == 1 else (dof_pos - self.default_dof_pos) * self.scales["dof_pos"]
        dof_vel_obs = (dof_vel * self.scales["dof_vel"]).unsqueeze(0) if dof_vel.ndim == 1 else dof_vel * self.scales["dof_vel"]
        last_act_obs = last_action if last_action.ndim == 2 else last_action.unsqueeze(0)
        last_last_act_obs = last_last_action if last_last_action.ndim == 2 else last_last_action.unsqueeze(0)

        # 3. 计算 4 维四足时钟输入
        clock_inputs = self.compute_clock_inputs(gait_indices, phase=0.5, offset=0.0, bound=0.0)

        # 4. 拼接标准 70 维观测
        obs = torch.cat([
            proj_grav_obs,        # 3
            scaled_commands,      # 15
            dof_pos_obs,          # 12
            dof_vel_obs,          # 12
            last_act_obs,         # 12
            last_last_act_obs,    # 12
            clock_inputs          # 4
        ], dim=-1) # Total = 70

        obs = torch.clamp(obs, -self.clip_observations, self.clip_observations)
        return obs

    def push_and_get_history(self, single_obs):
        """维护 [t-29, t-28, ..., t] 的 2100 维历史序列"""
        history_len = self.num_obs_history
        obs_dim = single_obs.shape[-1]
        
        if self._obs_history_flat is None or self._obs_history_flat.shape[-1] != history_len * obs_dim:
            self._obs_history_flat = single_obs.repeat(1, history_len)
        
        self._obs_history_flat = torch.cat([
            self._obs_history_flat[:, obs_dim:], 
            single_obs
        ], dim=-1)
        
        return self._obs_history_flat

    def reset_history(self, initial_single_obs):
        self._obs_history_flat = initial_single_obs.repeat(1, self.num_obs_history)
        return self._obs_history_flat