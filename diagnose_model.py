import os
import sys
import torch
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ──────────────────────────────────────────────
# 1. 加载 TorchScript 模型并打印结构
# ──────────────────────────────────────────────
MODEL_PATH = os.path.join(CURRENT_DIR, "assets", "body_latest.jit")
print(f"加载模型: {MODEL_PATH}")

model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

print("\n" + "=" * 60)
print("📋 模型结构:")
print("=" * 60)
print(model)

# 尝试获取模型代码
print("\n" + "=" * 60)
print("📋 模型 forward 代码:")
print("=" * 60)
try:
    print(model.code)
except:
    print("(无法获取 code，可能是内联模型)")

# ──────────────────────────────────────────────
# 2. 手工构造 "完美站立" 的 70 维观测
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("🧪 构造完美站立观测并推理")
print("=" * 60)

# --- obs_scales (与 observation_manager 中一致) ---
S_ANG_VEL = 0.25
S_LIN_VEL = 2.0
S_BODY_HEIGHT = 2.0
S_GAIT_FREQ = 1.0
S_GAIT_PHASE = 1.0
S_GAIT_OFFSET = 1.0
S_GAIT_BOUND = 1.0
S_GAIT_DURATION = 1.0
S_FOOTSWING = 0.15
S_BODY_PITCH = 0.3
S_BODY_ROLL = 0.3
S_STANCE_WIDTH = 1.0
S_STANCE_LENGTH = 1.0
S_AUX_REWARD = 1.0
S_DOF_POS = 1.0
S_DOF_VEL = 0.05

# --- 默认站立角度 ---
DEFAULT_ANGLES = np.array([
    0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,  # FR
    0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,  # RR
], dtype=np.float32)

# --- 默认 MoB 指令 ---
DEFAULT_BODY_HEIGHT = 0.30
DEFAULT_GAIT_FREQ = 3.0
DEFAULT_GAIT_PHASE = 0.5
DEFAULT_GAIT_OFFSET = 0.0
DEFAULT_GAIT_BOUND = 0.0
DEFAULT_GAIT_DURATION = 0.5
DEFAULT_FOOTSWING = 0.08
DEFAULT_BODY_PITCH = 0.0
DEFAULT_BODY_ROLL = 0.0
DEFAULT_STANCE_WIDTH = 0.275
DEFAULT_STANCE_LENGTH = 0.25
DEFAULT_AUX_REWARD = 0.0

def build_perfect_obs():
    """构造一个完美站立状态下的 70 维观测"""
    obs = np.zeros(70, dtype=np.float32)
    
    # [0:3] base_ang_vel * scale (静止，角速度为 0)
    obs[0:3] = np.array([0.0, 0.0, 0.0]) * S_ANG_VEL
    
    # [3:6] projected_gravity (重力沿 -z，机身水平时为 [0, 0, -1])
    obs[3:6] = np.array([0.0, 0.0, -1.0])
    
    # [6:21] commands * commands_scale (15 维)
    raw_cmd = np.array([
        0.0, 0.0, 0.0,                          # vx, vy, wz (静止)
        DEFAULT_BODY_HEIGHT,                     # body_height
        DEFAULT_GAIT_FREQ,                        # gait_freq
        DEFAULT_GAIT_PHASE,                       # gait_phase
        DEFAULT_GAIT_OFFSET,                      # gait_offset
        DEFAULT_GAIT_BOUND,                       # gait_bound
        DEFAULT_GAIT_DURATION,                    # gait_duration
        DEFAULT_FOOTSWING,                        # footswing_height
        DEFAULT_BODY_PITCH,                       # body_pitch
        DEFAULT_BODY_ROLL,                        # body_roll
        DEFAULT_STANCE_WIDTH,                     # stance_width
        DEFAULT_STANCE_LENGTH,                    # stance_length
        DEFAULT_AUX_REWARD,                       # aux_reward
    ], dtype=np.float32)
    cmd_scale = np.array([
        S_LIN_VEL, S_LIN_VEL, S_ANG_VEL,
        S_BODY_HEIGHT, S_GAIT_FREQ, S_GAIT_PHASE, S_GAIT_OFFSET,
        S_GAIT_BOUND, S_GAIT_DURATION,
        S_FOOTSWING, S_BODY_PITCH, S_BODY_ROLL,
        S_STANCE_WIDTH, S_STANCE_LENGTH, S_AUX_REWARD
    ], dtype=np.float32)
