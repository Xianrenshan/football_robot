import os

# ============================================================
#               Go1 预训练策略与系统基础配置
# ============================================================

# 1. 资源路径定义（以项目根目录为基准锚定）
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CONFIG_DIR)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")

BODY_MODEL_PATH = os.path.join(ASSETS_DIR, "body_latest.jit")
ADAPTATION_MODEL_PATH = os.path.join(ASSETS_DIR, "adaptation_module_latest.jit")
PARAMS_PATH = os.path.join(ASSETS_DIR, "parameters.pkl")
POLICY_CONFIG_PATH = os.path.join(ASSETS_DIR, "policy_config.json")
SCENE_XML_PATH = os.path.join(PROJECT_ROOT, "field_scene.xml")

# 2. 控制时钟与分频
SIM_DT = 0.002                            # MuJoCo 物理积分步长 (500Hz)
POLICY_FREQ = 50.0                        # 策略决策频率 (50Hz)
POLICY_DT = 1.0 / POLICY_FREQ             # 决策周期 (0.02s)
DECIMATION = int(round(POLICY_DT / SIM_DT)) # 分频系数 (10 步)

# 3. 运控安全限幅
MAX_VX = 1.5                              # 最大前进线速度 (m/s)
MIN_VX = -0.8                             # 最大后退线速度 (m/s)
MAX_VY = 0.8                              # 最大横向线速度 (m/s)
MAX_WZ = 2.0                              # 最大自旋角速度 (rad/s)
MAX_TORQUE = 23.7                         # Go1 电机最大额定力矩 (N·m)

# 4. 标称物理与步态参数 (Walk-These-Ways MoB 标准)
NOMINAL_STAND_HEIGHT = 0.28               # 标称站立落地点躯干高度 (m)
DEFAULT_BODY_HEIGHT = 0.0                 # 机身高度调节偏移 (m)
DEFAULT_GAIT_FREQ = 3.0                   # 默认对角小跑步态频率 (Hz)
DEFAULT_GAIT_PHASE = 0.5                  # 步态相位差 (0.5 为对角 Trot)
DEFAULT_GAIT_OFFSET = 0.0                 # 步态偏移
DEFAULT_GAIT_BOUND = 0.0                  # 步态边界
DEFAULT_GAIT_DURATION = 0.5               # 支撑相比例 (0.5 为标准对角小跑)
DEFAULT_FOOTSWING_HEIGHT = 0.08           # 抬腿高度 (m)
DEFAULT_BODY_PITCH = 0.0                  # 机身俯仰偏角 (rad)
DEFAULT_BODY_ROLL = 0.0                   # 机身横滚偏角 (rad)
DEFAULT_STANCE_WIDTH = 0.0                # 站立宽度偏移 (m)
DEFAULT_STANCE_LENGTH = 0.0               # 站立长度偏移 (m)
DEFAULT_AUX_REWARD = 0.0                  # 辅助偏置

# 5. 执行器默认阻抗增益
DEFAULT_KP = 20.0                         # 关节比例刚度
DEFAULT_KD = 0.5                          # 关节阻尼
ACTION_SCALE = 0.25                       # 策略动作缩放系数