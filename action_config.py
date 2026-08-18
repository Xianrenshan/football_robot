import os

# ============================================================
#               Go1 预训练策略与运动参数配置
# ============================================================

# 1. 资源与模型路径配置
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(CURRENT_DIR, "assets")

BODY_MODEL_PATH = os.path.join(ASSETS_DIR, "body_latest.jit")
ADAPTATION_MODEL_PATH = os.path.join(ASSETS_DIR, "adaptation_module_latest.jit")
PARAMS_PATH = os.path.join(ASSETS_DIR, "parameters.pkl")

# 2. 控制时钟与分频配置 (50Hz 策略决策)
POLICY_FREQ = 50.0                        # 策略决策频率 (Hz)
POLICY_DT = 1.0 / POLICY_FREQ             # 决策周期 (0.02 秒)

# 3. 速度指令安全限幅
MAX_VX = 1.5                              # 最大前进线速度 (m/s)
MIN_VX = -0.8                             # 最大后退线速度 (m/s)
MAX_VY = 0.8                              # 最大横向移动速度 (m/s)
MAX_WZ = 2.0                              # 最大自旋角速度 (rad/s)

# 4. Walk-These-Ways (MoB) 默认步态与姿态参数
DEFAULT_BODY_HEIGHT = 0.0                 # 机身名义高度偏移 (m)
DEFAULT_GAIT_FREQ = 3.0                   # 默认步态频率 (Hz)
DEFAULT_GAIT_PHASE = 0.5                  # 步态相位差 (对角小跑)
DEFAULT_GAIT_OFFSET = 0.0                 # 步态偏移
DEFAULT_GAIT_BOUND = 0.0                  # 步态边界
DEFAULT_GAIT_DURATION = 0.5               # 支撑相比例 (0.5 为对角小跑)
DEFAULT_FOOTSWING_HEIGHT = 0.08           # 摆动腿抬腿高度 (m)
DEFAULT_BODY_PITCH = 0.0                  # 俯仰偏角 (rad)
DEFAULT_BODY_ROLL = 0.0                   # 横滚偏角 (rad)
DEFAULT_STANCE_WIDTH = 0.0                # 站立宽度偏移 (m)
DEFAULT_STANCE_LENGTH = 0.0               # 站立长度偏移 (m)
DEFAULT_AUX_REWARD = 0.0                  # 辅助偏置

# 5. 策略执行器默认回退标定参数 (当参数文件缺省时使用)
DEFAULT_KP = 20.0                         # 关节比例刚度
DEFAULT_KD = 0.5                          # 关节阻尼
ACTION_SCALE = 0.25                       # 策略输出动作缩放因子
MAX_TORQUE = 23.7                         # 电机最大输出力矩 (N·m)