import os
import sys
import time

# 动态确保项目根目录在 sys.path 中，杜绝任何外部目录启动导致的导入失败
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import mujoco
import mujoco.viewer
import numpy as np
from config.action_config import SCENE_XML_PATH, NOMINAL_STAND_HEIGHT
from core.motion_controller import Go1MotionController
from camera_controller import ThirdPersonCameraController
from behaviors.soccer_agent import SoccerAgent

# 全局控制指令与状态标志
keyboard_cmd = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "cam_mode": 1,         # 1: 第三人称平滑追随, 2: 全场沙盘, 3: 球门特写
    "reset_requested": False
}

agent = SoccerAgent(goal_pos=np.array([4.4, 0.0]))

def key_callback(keycode):
    """键盘交互监听"""
    char = chr(keycode).upper() if 0 <= keycode < 128 else ""

    # 前进 / 后退
    if char == "W":
        keyboard_cmd["vx"] = min(keyboard_cmd["vx"] + 0.25, 1.2)
    elif char == "S":
        keyboard_cmd["vx"] = max(keyboard_cmd["vx"] - 0.25, -0.8)

    # 原地转向
    elif char == "A":
        keyboard_cmd["wz"] = min(keyboard_cmd["wz"] + 0.35, 1.5)
    elif char == "D":
        keyboard_cmd["wz"] = max(keyboard_cmd["wz"] - 0.35, -1.5)

    # 横移
    elif char == "Q":
        keyboard_cmd["vy"] = min(keyboard_cmd["vy"] + 0.2, 0.6)
    elif char == "E":
        keyboard_cmd["vy"] = max(keyboard_cmd["vy"] - 0.2, -0.6)

    # 驻车制动 (急停归零)
    elif char == "X" or keycode == 32:  # X 键或空格键
        keyboard_cmd["vx"] = 0.0
        keyboard_cmd["vy"] = 0.0
        keyboard_cmd["wz"] = 0.0
        print("[控制] 速度归零 -> 自平衡站立锁定")

    # 自动踢球战术切换
    elif char == "M":
        is_auto = agent.toggle_auto_mode()
        status = "【开启】AI 自动追球与射门" if is_auto else "【关闭】已切换至手动键盘控制"
        print(f"[模式] {status}")

    # 相机视角切换
    elif char == "1":
        keyboard_cmd["cam_mode"] = 1
        print("[相机] 切换 -> 第三人称追随视角")
    elif char == "2":
        keyboard_cmd["cam_mode"] = 2
        print("[相机] 切换 -> 全场俯瞰视角")
    elif char == "3":
        keyboard_cmd["cam_mode"] = 3
        print("[相机] 切换 -> 正对球门特写视角")

    # 场景重置
    elif char == "R":
        keyboard_cmd["reset_requested"] = True
        print("[系统] 重置机器人与足球状态")
def reset_simulation_state(model, data, controller):
    """将机器人平稳放置于地面并重置控制器状态"""
    mujoco.mj_resetData(model, data)

    # 1. 设置 12 关节标称站立角度
    for jname, q_val in zip(controller.policy_joint_names, controller.default_dof_pos.numpy()):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = q_val

    # 2. 设置躯干初始高度 (标称高度 0.34m) 与姿态
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id != -1:
        trunk_jnt = model.body_jntadr[trunk_id]
        if trunk_jnt != -1:
            q_adr = model.jnt_qposadr[trunk_jnt]
            data.qpos[q_adr: q_adr + 3] = [0.0, 0.0, 0.34]
            data.qpos[q_adr + 3: q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    # 3. 控制器重置
    controller.reset_state()

def main():
    if not os.path.exists(SCENE_XML_PATH):
        raise FileNotFoundError(f"未找到场景 XML: {SCENE_XML_PATH}")

    print(f"正在加载场景: {SCENE_XML_PATH} ...")
    model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    # 实例化控制器与相机
    controller = Go1MotionController(model, data)
    cam_controller = ThirdPersonCameraController(
        distance=1.8,
        elevation_deg=-20.0,
        lookat_height_offset=0.30,
        smoothness_azimuth=0.12,
        smoothness_elevation=0.12
    )

    reset_simulation_state(model, data, controller)

    robot_trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")

    print("\n" + "="*60)
    print(" 🎮 Go1 足球机器人仿真控制台已就绪！（分层强化学习架构）")
    print(" ────────────────────────────────────────────────────────")
    print("  [W / S]   : 前进 / 后退")
    print("  [A / D]   : 原地左转 / 原地右转")
    print("  [Q / E]   : 向左侧移 / 向右侧移")
    print("  [X / 空格]: 刹车驻车（原地稳固自平衡站立）")
    print("  [M]       : 切换【自动追球射门】/【手动驾驶】模式")
    print(" ────────────────────────────────────────────────────────")
    print("  [1/2/3]   : 视角切换 (第三人称跟随 / 沙盘 / 球门)")
    print("  [R]       : 重置机器人与足球位姿")
    print("="*60 + "\n")

    TARGET_RENDER_FPS = 60.0
    FRAME_INTERVAL = 1.0 / TARGET_RENDER_FPS
    steps_per_frame = max(1, int(round(FRAME_INTERVAL / model.opt.timestep)))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_print_time = time.time()

        while viewer.is_running():
            # 处理场景重置
            if keyboard_cmd["reset_requested"]:
                reset_simulation_state(model, data, controller)
                keyboard_cmd["vx"] = 0.0
                keyboard_cmd["vy"] = 0.0
                keyboard_cmd["wz"] = 0.0
                keyboard_cmd["reset_requested"] = False
                cam_controller.reset()

            # 战术层决策介入（如果开启了自动模式）
            robot_pos = data.xpos[robot_trunk_id]
            robot_quat = data.xquat[robot_trunk_id]
            ball_pos = data.xpos[ball_body_id]

            auto_cmd = agent.update_decision(robot_pos, robot_quat, ball_pos)
            if auto_cmd is not None:
                cmd_vx, cmd_vy, cmd_wz = auto_cmd
            else:
                cmd_vx, cmd_vy, cmd_wz = keyboard_cmd["vx"], keyboard_cmd["vy"], keyboard_cmd["wz"]

            # 下发指令给运控层
            controller.set_velocity_command(cmd_vx, cmd_vy, cmd_wz)

            # 500Hz 高频物理步进与分频策略推理
            for _ in range(steps_per_frame):
                controller.update()
                mujoco.mj_step(model, data)

            # 相机视角追踪更新
            if keyboard_cmd["cam_mode"] == 1:
                az_deg, el_deg, lookat = cam_controller.update(robot_pos, robot_quat)
                viewer.cam.lookat[:] = lookat
                viewer.cam.distance = cam_controller.distance
                viewer.cam.azimuth = az_deg
                viewer.cam.elevation = el_deg
            elif keyboard_cmd["cam_mode"] == 2:
                viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
                viewer.cam.distance = 7.5
                viewer.cam.elevation = -45.0
                viewer.cam.azimuth = 90.0
            elif keyboard_cmd["cam_mode"] == 3:
                viewer.cam.lookat[:] = [4.4, 0.0, 0.5]
                viewer.cam.distance = 3.5
                viewer.cam.elevation = -15.0
                viewer.cam.azimuth = 180.0

            viewer.sync()

            # 控制台遥测监控
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                dist = np.linalg.norm(robot_pos[:2] - ball_pos[:2])
                mode_str = f"AUTO({agent.state})" if agent.auto_mode else "MANUAL"
                print(f"[{mode_str:<15}] 机身坐标: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) | "
                      f"速度指令: (vx={controller.cmd_vx:.2f}, vy={controller.cmd_vy:.2f}, wz={controller.cmd_wz:.2f}) | "
                      f"距球: {dist:.2f}m")
                last_print_time = current_time

if __name__ == "__main__":
    main()