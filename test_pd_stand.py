import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

# 确保项目根目录在 sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config.action_config import (
    SCENE_XML_PATH, NOMINAL_STAND_HEIGHT,
    DEFAULT_KP, DEFAULT_KD, MAX_TORQUE
)

# ──────────────────────────────────────────────
# 1. 标准 12 关节排布顺序 (与 Go1MotionController 一致)
# ──────────────────────────────────────────────
POLICY_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"
]

DEFAULT_ANGLES = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}

DEFAULT_DOF_POS = np.array(
    [DEFAULT_ANGLES[j] for j in POLICY_JOINT_NAMES], dtype=np.float64
)

KP = DEFAULT_KP   # 20.0
KD = DEFAULT_KD   # 0.5


def build_joint_indices(model):
    """构建 qpos / qvel / ctrl 索引，逻辑与 Go1MotionController 完全一致"""
    qpos_idx, qvel_idx, ctrl_idx = [], [], []

    for jname in POLICY_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid == -1:
            raise ValueError(f"模型中找不到关节: {jname}")
        qpos_idx.append(model.jnt_qposadr[jid])
        qvel_idx.append(model.jnt_dofadr[jid])

    actuator_names = [
        "FL_hip", "FL_thigh", "FL_calf",
        "FR_hip", "FR_thigh", "FR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
    ]
    for aname in actuator_names:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid != -1:
            ctrl_idx.append(aid)
        else:
            ctrl_idx = list(range(12))
            break

    return (
        np.array(qpos_idx, dtype=np.int32),
        np.array(qvel_idx, dtype=np.int32),
        np.array(ctrl_idx, dtype=np.int32),
    )


def reset_robot(model, data, qpos_idx):
    """将机器人放置到标称站立姿态"""
    mujoco.mj_resetData(model, data)

    # 设置 trunk 高度和朝向
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id != -1:
        trunk_jnt = model.body_jntadr[trunk_id]
        if trunk_jnt != -1:
            q_adr = model.jnt_qposadr[trunk_jnt]
            data.qpos[q_adr: q_adr + 3] = [0.0, 0.0, NOMINAL_STAND_HEIGHT]
            data.qpos[q_adr + 3: q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    # 设置 12 关节标称角度
    for jname, q_val in zip(POLICY_JOINT_NAMES, DEFAULT_DOF_POS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = q_val

    mujoco.mj_forward(model, data)


def main():
    if not os.path.exists(SCENE_XML_PATH):
        raise FileNotFoundError(f"未找到场景 XML: {SCENE_XML_PATH}")

    print(f"正在加载场景: {SCENE_XML_PATH}")
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    qpos_idx, qvel_idx, ctrl_idx = build_joint_indices(model)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")

    reset_robot(model, data, qpos_idx)

    print("=" * 60)
    print(" 🔧 PD 站立测试（无神经网络策略）")
    print(" 目标: 验证执行器映射 & PD 增益能否维持标称站立")
    print(f" Kp={KP}, Kd={KD}, StandHeight={NOMINAL_STAND_HEIGHT}m")
    print(" 按 [R] 重置 | 关闭窗口退出")
    print("=" * 60)

    # 打印执行器映射表，供人工核对
    print("\n📋 执行器映射表:")
    print(f"  {'关节名称':<20} | {'qpos_idx':<10} | {'qvel_idx':<10} | {'ctrl_idx':<10} | {'默认角度':<10}")
    print("  " + "-" * 70)
    for i, jname in enumerate(POLICY_JOINT_NAMES):
        print(f"  {jname:<20} | {qpos_idx[i]:<10} | {qvel_idx[i]:<10} | {ctrl_idx[i]:<10} | {DEFAULT_DOF_POS[i]:<10.2f}")
    print()

    TARGET_FPS = 60.0
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    steps_per_frame = max(1, int(round(FRAME_INTERVAL / model.opt.timestep)))

    reset_flag = {"requested": False}

    def key_callback(keycode):
        char = chr(keycode).upper() if 0 <= keycode < 128 else ""
        if char == "R":
            reset_flag["requested"] = True
            print("[系统] 重置机器人状态")

    last_print = time.time()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            if reset_flag["requested"]:
                reset_robot(model, data, qpos_idx)
                reset_flag["requested"] = False

            # ── 纯 PD 闭环：目标永远是标称站立角度 ──
            for _ in range(steps_per_frame):
                current_q = data.qpos[qpos_idx]
                current_dq = data.qvel[qvel_idx]

                torques = KP * (DEFAULT_DOF_POS - current_q) - KD * current_dq
                torques = np.clip(torques, -MAX_TORQUE, MAX_TORQUE)

                for i, aid in enumerate(ctrl_idx):
                    data.ctrl[aid] = torques[i]

                mujoco.mj_step(model, data)

            viewer.sync()

            # 遥测
            now = time.time()
            if now - last_print >= 0.5:
                trunk_pos = data.xpos[trunk_id]
                q_err = np.abs(data.qpos[qpos_idx] - DEFAULT_DOF_POS)
                max_err = np.max(q_err)
                print(
                    f"[PD-TEST] Z={trunk_pos[2]:.3f}m | "
                    f"最大关节偏差={np.degrees(max_err):.1f}° | "
                    f"力矩范围=[{torques.min():.1f}, {torques.max():.1f}] N·m"
                )
                last_print = now


if __name__ == "__main__":
    main()
