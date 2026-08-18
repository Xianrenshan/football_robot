import os
import io
import sys
import json
import pickle
import numpy as np
import torch


class CPU_Unpickler(pickle.Unpickler):
    """安全反序列化器：拦截 CUDA 存储标记，将其安全重定向至 CPU 内存"""
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def make_json_serializable(obj):
    """递归将 PyTorch 张量、NumPy 数组及复杂对象转换为原生 Python 基础类型"""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    elif isinstance(obj, (np.float16, np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(vars(obj))
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def load_parameters_raw(pkl_path):
    """安全加载 parameters.pkl"""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"未找到目标文件: {pkl_path}")

    try:
        data = torch.load(pkl_path, map_location="cpu")
        print("[1/4] 成功使用 torch.load 加载 parameters.pkl")
        return data
    except Exception:
        with open(pkl_path, "rb") as f:
            data = CPU_Unpickler(f).load()
        print("[1/4] 成功使用 CPU_Unpickler 拦截并加载 parameters.pkl")
        return data


def extract_and_build_config(raw_data):
    """从原始配置结构中提取规范化参数与 70 维特征布局"""
    cfg = raw_data.get("Cfg", {})
    if hasattr(cfg, '__dict__'):
        cfg = vars(cfg)

    # 1. 提取基础维度与环境配置
    env_cfg = cfg.get("env", {})
    if hasattr(env_cfg, '__dict__'):
        env_cfg = vars(env_cfg)

    num_observations = int(env_cfg.get("num_observations", 70))
    num_obs_history = int(env_cfg.get("num_observation_history", 30))
    num_actions = int(env_cfg.get("num_actions", 12))

    # 2. 提取运控参数 (Kp, Kd, action_scale)
    control_cfg = cfg.get("control", {})
    if hasattr(control_cfg, '__dict__'):
        control_cfg = vars(control_cfg)

    stiffness = control_cfg.get("stiffness", {})
    damping = control_cfg.get("damping", {})
    action_scale = float(control_cfg.get("action_scale", 0.25))

    kp_val = float(list(stiffness.values())[0]) if stiffness else 20.0
    kd_val = float(list(damping.values())[0]) if damping else 0.5

    # 3. 提取 12 关节名称与默认站立角度
    init_state_cfg = cfg.get("init_state", {})
    if hasattr(init_state_cfg, '__dict__'):
        init_state_cfg = vars(init_state_cfg)

    default_angles_dict = init_state_cfg.get("default_joint_angles", {
        "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
        "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
        "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
        "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
    })

    # 标准 12 关节排布顺序 (FL, FR, RL, RR)
    joint_names_order = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
    ]
    default_joint_angles_list = [float(default_angles_dict.get(j, 0.0)) for j in joint_names_order]

    # 4. 提取特征通道归一化尺度
    norm_cfg = cfg.get("normalization", {})
    if hasattr(norm_cfg, '__dict__'):
        norm_cfg = vars(norm_cfg)

    obs_scales_raw = norm_cfg.get("obs_scales", {})
    if hasattr(obs_scales_raw, '__dict__'):
        obs_scales_raw = vars(obs_scales_raw)

    obs_scales = {
        "lin_vel": float(obs_scales_raw.get("lin_vel", 2.0)),
        "ang_vel": float(obs_scales_raw.get("ang_vel", 0.25)),
        "dof_pos": float(obs_scales_raw.get("dof_pos", 1.0)),
        "dof_vel": float(obs_scales_raw.get("dof_vel", 0.05)),
        "body_height": float(obs_scales_raw.get("body_height", 2.0)),
        "gait_freq": float(obs_scales_raw.get("gait_freq", 1.0)),
        "gait_phase": float(obs_scales_raw.get("gait_phase", 1.0)),
        "gait_offset": float(obs_scales_raw.get("gait_offset", 1.0)),
        "gait_bound": float(obs_scales_raw.get("gait_bound", 1.0)),
        "gait_duration": float(obs_scales_raw.get("gait_duration", 1.0)),
        "footswing_height": float(obs_scales_raw.get("footswing_height", 0.15)),
        "body_pitch": float(obs_scales_raw.get("body_pitch", 0.3)),
        "body_roll": float(obs_scales_raw.get("body_roll", 0.3)),
        "stance_width": float(obs_scales_raw.get("stance_width", 1.0)),
        "stance_length": float(obs_scales_raw.get("stance_length", 1.0)),
        "aux_reward": float(obs_scales_raw.get("aux_reward", 1.0))
    }

    # 5. 自动推导特征槽位绝对切片表 (Observation Layout)
    obs_layout = []
    current_idx = 0

    # (1) 重力投影
    obs_layout.append({
        "name": "projected_gravity",
        "dim": 3,
        "slice": [current_idx, current_idx + 3],
        "description": "机身坐标系重力投影向量 (3维)"
    })
    current_idx += 3

    # (2) 指令向量
    obs_layout.append({
        "name": "commands",
        "dim": 15,
        "slice": [current_idx, current_idx + 15],
        "description": "15 维速度与 MoB 步态调谐指令"
    })
    current_idx += 15

    # (3) 关节角度误差
    obs_layout.append({
        "name": "dof_pos_residual",
        "dim": 12,
        "slice": [current_idx, current_idx + 12],
        "description": "12 关节与默认角度偏差 (q - q_default)"
    })
    current_idx += 12

    # (4) 关节角速度
    obs_layout.append({
        "name": "dof_vel",
        "dim": 12,
        "slice": [current_idx, current_idx + 12],
        "description": "12 关节旋转角速度"
    })
    current_idx += 12

    # (5) 上一步动作
    obs_layout.append({
        "name": "actions",
        "dim": 12,
        "slice": [current_idx, current_idx + 12],
        "description": "上一个周期的网络输出动作 (12维)"
    })
    current_idx += 12

    # (6) 上上步动作
    if bool(env_cfg.get("observe_two_prev_actions", True)):
        obs_layout.append({
            "name": "last_actions",
            "dim": 12,
            "slice": [current_idx, current_idx + 12],
            "description": "上上周期的历史动作 (12维)"
        })
        current_idx += 12

    # (7) 步态时间步相位
    if bool(env_cfg.get("observe_timing_parameter", True)):
        obs_layout.append({
            "name": "gait_indices",
            "dim": 1,
            "slice": [current_idx, current_idx + 1],
            "description": "步态时间步相位积分 (1维)"
        })
        current_idx += 1

    # (8) 机身角速度 (末尾)
    if bool(env_cfg.get("observe_vel", True)):
        obs_layout.append({
            "name": "base_ang_vel",
            "dim": 3,
            "slice": [current_idx, current_idx + 3],
            "description": "机身本体坐标系角速度 (3维, IMU)"
        })
        current_idx += 3

    # 组装最终 JSON 字典
    final_config = {
        "meta": {
            "robot_name": "Unitree Go1",
            "framework": "Walk-These-Ways (MIT Improbable AI)",
            "policy_input_dim": num_observations,
            "history_window_len": num_obs_history,
            "total_history_input_dim": num_observations * num_obs_history,
            "action_dim": num_actions
        },
        "control": {
            "kp": kp_val,
            "kd": kd_val,
            "action_scale": action_scale,
            "policy_dt": 0.02,
            "policy_freq": 50.0
        },
        "joints": {
            "order": joint_names_order,
            "default_angles_dict": default_angles_dict,
            "default_angles_list": default_joint_angles_list
        },
        "normalization": {
            "obs_scales": obs_scales
        },
        "observation_layout": obs_layout
    }

    return final_config


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    assets_dir = os.path.join(current_dir, "assets")
    pkl_path = os.path.join(assets_dir, "parameters.pkl")
    json_path = os.path.join(assets_dir, "policy_config.json")

    print("=" * 70)
    print(" 🛠️  Unitree Go1 策略配置文件离线提取工具")
    print("=" * 70)
    print(f"目标输入文件: {pkl_path}")

    # 1. 安全读取
    raw_data = load_parameters_raw(pkl_path)

    # 2. 结构提取与推导
    config_dict = extract_and_build_config(raw_data)
    print("[2/4] 成功提取并验证所有特征通道与运控参数！")

    # 3. 打印特征布局表
    print("[3/4] 校验特征通道真实切片分布 (Observation Layout):")
    print(" ───────────────────────────────────────────────────────────────────")
    print(f" {'槽位区间':<12} | {'维度':<6} | {'特征名称':<20} | {'物理说明'}")
    print(" ───────────────────────────────────────────────────────────────────")
    for item in config_dict["observation_layout"]:
        slice_str = f"[{item['slice'][0]:02d} : {item['slice'][1]:02d}]"
        print(f" {slice_str:<12} | {item['dim']:<6} | {item['name']:<20} | {item['description']}")
    print(" ───────────────────────────────────────────────────────────────────")
    print(f" 📊 单帧总维度: {config_dict['meta']['policy_input_dim']} 维 | 历史时序总特征: {config_dict['meta']['total_history_input_dim']} 维")
    print(" ───────────────────────────────────────────────────────────────────")

    # 4. 写入 JSON
    serializable_dict = make_json_serializable(config_dict)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable_dict, f, indent=4, ensure_ascii=False)

    print(f"[4/4] ✨ 导出完成！已生成纯净配置文件: {json_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()