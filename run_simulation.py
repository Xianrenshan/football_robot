import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from controller import Go1MotionController
from camera_controller import ThirdPersonCameraController

# 全局键盘控制与状态缓存
keyboard_cmd = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "cam_mode": 1,         # 1: 第三人称追随, 2: 全场俯瞰, 3: 球门特写
    "reset_requested": False
}


def key_callback(keycode):
    """键盘交互事件监听器：连续速度调谐与即走即停"""
    char = chr(keycode).upper() if 0 <= keycode < 128 else ""

    # 前进 / 后退
    if char == "W":
        keyboard_cmd["vx"] = min(keyboard_cmd["vx"] + 0.3, 1.2)
    elif char == "S":
        keyboard_cmd["vx"] = max(keyboard_cmd["vx"] - 0.3, -0.8)

    # 左右自旋
    elif char == "A":
        keyboard_cmd["wz"] = min(keyboard_cmd["wz"] + 0.4, 1.5)
    elif char == "D":
        keyboard_cmd["wz"] = max(keyboard_cmd["wz"] - 0.4, -1.5)

    # 左右侧移
    elif char == "Q":
        keyboard_cmd["vy"] = min(keyboard_cmd["vy"] + 0.2, 0.6)
    elif char == "E":
        keyboard_cmd["vy"] = max(keyboard_cmd["vy"] - 0.2, -0.6)

    # 急停 / 刹车：直接归零
    elif char == "X" or keycode == 32:  # X 键或空格键
        keyboard_cmd["vx"] = 0.0
        keyboard_cmd["vy"] = 0.0
        keyboard_cmd["wz"] = 0.0
        print("[控制] 速度指令归零 -> 进入自平衡站立锁定")

    # 相机视角切换
    elif char == "1":
        keyboard_cmd["cam_mode"] = 1
        print("[相机] 切换 -> 第三人称动态跟随视角")
    elif char == "2":
        keyboard_cmd["cam_mode"] = 2
        print("[相机] 切换 -> 全场俯瞰沙盘视角")
    elif char == "3":
        keyboard_cmd["cam_mode"] = 3
        print("[相机] 切换 -> 正对球门特写视角")

    # 场景重置
    elif char == "R":
        keyboard_cmd["reset_requested"] = True
        print("[系统] 重置机器人与足球状态")


def reset_simulation_state(model, data, controller):
    """重置机器人位姿、足球位置及策略控制器状态"""
    mujoco.mj_resetData(model, data)

    # 1. 设置机身初始位置与姿态 (Z = 0.36m，贴合策略名义站立高度)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id != -1:
        trunk_jnt = model.body_jntadr[trunk_id]
        if trunk_jnt != -1:
            q_adr = model.jnt_qposadr[trunk_jnt]
            data.qpos[q_adr : q_adr + 3] = [0.0, 0.0, 0.36]
            data.qpos[q_adr + 3 : q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    # 2. 设置 12 关节为策略默认标定角度
    for jname, q_val in zip(controller.policy_joint_names, controller.default_dof_pos.numpy()):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = q_val

    # 3. 设置足球初始落点 (X = 1.2m, Z = 0.11m)
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
    if ball_id != -1:
        ball_jnt = model.body_jntadr[ball_id]
        if ball_jnt != -1:
            b_adr = model.jnt_qposadr[ball_jnt]
            data.qpos[b_adr : b_adr + 3] = [1.2, 0.0, 0.11]
            data.qpos[b_adr + 3 : b_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

    # 4. 重置策略内部状态并预运行一帧
    controller.reset_state()
    controller.update()


def main():
    global controller

    # 1. 模型路径解析与加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "field_scene.xml")

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"找不到场景文件: {scene_path}")

    print(f"正在加载场景: {scene_path} ...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 2. 初始化预训练策略控制器与第三人称相机
    controller = Go1MotionController(model, data)

    cam_controller = ThirdPersonCameraController(
        distance=1.8,
        elevation_deg=-20.0,
        lookat_height_offset=0.35,
        smoothness_azimuth=0.10,
        smoothness_elevation=0.10
    )

    # 3. 初始化仿真位姿
    reset_simulation_state(model, data, controller)

    robot_trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")

    print("\n" + "="*60)
    print(" 🎮 Go1 足球机器人仿真控制台已就绪！（RL 预训练策略模式）")
    print(" ────────────────────────────────────────────────────────")
    print("  [W / S]   : 前进 / 后退")
    print("  [A / D]   : 原地左转 / 原地右转")
    print("  [Q / E]   : 向左横移 / 向右横移")
    print("  [X / 空格]: 刹车驻车（指令归零，原地稳固自平衡）")
    print(" ────────────────────────────────────────────────────────")
    print("  [1/2/3]   : 切换视角")
    print("  [R]       : 重置场景")
    print("="*60 + "\n")

    # 4. 分频渲染与物理步进主循环
    TARGET_RENDER_FPS = 60.0
    FRAME_INTERVAL = 1.0 / TARGET_RENDER_FPS
    steps_per_frame = max(1, int(round(FRAME_INTERVAL / model.opt.timestep)))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_print_time = time.time()

        while viewer.is_running():
            # 处理场景重置请求
            if keyboard_cmd["reset_requested"]:
                reset_simulation_state(model, data, controller)
                keyboard_cmd["vx"] = 0.0
                keyboard_cmd["vy"] = 0.0
                keyboard_cmd["wz"] = 0.0
                keyboard_cmd["reset_requested"] = False
                cam_controller.reset()

            # 将键盘指令下发给控制器
            controller.set_velocity_command(
                keyboard_cmd["vx"],
                keyboard_cmd["vy"],
                keyboard_cmd["wz"]
            )

            # 高频动力学步进
            for _ in range(steps_per_frame):
                controller.update()
                mujoco.mj_step(model, data)

            # 动态相机追踪更新
            if keyboard_cmd["cam_mode"] == 1:
                robot_pos = data.xpos[robot_trunk_id]
                robot_quat = data.xquat[robot_trunk_id]
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

            # 终端状态监控打印
            current_time = time.time()
            if current_time - last_print_time >= 0.5:
                ball_pos = data.xpos[ball_body_id]
                robot_pos = data.xpos[robot_trunk_id]
                dist = np.linalg.norm(robot_pos[:2] - ball_pos[:2])
                print(f"[机身] 坐标: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) | "
                      f"目标速度: (vx={controller.cmd_vx:.2f}, vy={controller.cmd_vy:.2f}, wz={controller.cmd_wz:.2f}) | "
                      f"距球: {dist:.2f} m")
                last_print_time = current_time


if __name__ == "__main__":
    main()