import os
import time
import mujoco
import mujoco.viewer
import numpy as np
from controller import Go1MotionController
from camera_controller import ThirdPersonCameraController 

# 全局键盘控制指令缓存
keyboard_cmd = {
    "vx": 0.0,
    "vy": 0.0,
    "wz": 0.0,
    "jump_triggered": False,
    "cam_mode": 1,         # 1: 第三人称追随, 2: 全场俯瞰, 3: 球门特写
    "reset_requested": False
}

def key_callback(keycode):
    """键盘交互事件监听器"""
    LIN_STEP = 0.25
    ANG_STEP = 0.40

    char = chr(keycode).upper() if 0 <= keycode < 128 else ""

    if char == "W":
        keyboard_cmd["vx"] = min(keyboard_cmd["vx"] + LIN_STEP, 1.0)
    elif char == "S":
        keyboard_cmd["vx"] = max(keyboard_cmd["vx"] - LIN_STEP, -0.6)
    elif char == "A":
        keyboard_cmd["wz"] = min(keyboard_cmd["wz"] + ANG_STEP, 1.2)
    elif char == "D":
        keyboard_cmd["wz"] = max(keyboard_cmd["wz"] - ANG_STEP, -1.2)
    elif char == "Q":
        keyboard_cmd["vy"] = min(keyboard_cmd["vy"] + LIN_STEP, 0.4)
    elif char == "E":
        keyboard_cmd["vy"] = max(keyboard_cmd["vy"] - LIN_STEP, -0.4)
    elif char == "X":
        # 紧急制动
        keyboard_cmd["vx"] = 0.0
        keyboard_cmd["vy"] = 0.0
        keyboard_cmd["wz"] = 0.0
    elif keycode == 32:  # 空格键
        keyboard_cmd["jump_triggered"] = True
    elif char == "1":
        keyboard_cmd["cam_mode"] = 1
        print("[相机] 切换 -> 第三人称动态跟随视角")
    elif char == "2":
        keyboard_cmd["cam_mode"] = 2
        print("[相机] 切换 -> 全场俯瞰沙盘视角")
    elif char == "3":
        keyboard_cmd["cam_mode"] = 3
        print("[相机] 切换 -> 正对球门特写视角")
    elif char == "R":
        keyboard_cmd["reset_requested"] = True
        print("[系统] 重置机器人与足球状态")

def reset_simulation_state(model, data, controller):
    """动态安全重置机器人位姿与足球位置"""
    mujoco.mj_resetData(model, data)

    # 1. 动态定位机身浮动基座并设置初始悬空站立位姿 (Z = 0.36m)
    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    if trunk_id != -1:
        trunk_jnt = model.body_jntadr[trunk_id]
        if trunk_jnt != -1:
            q_adr = model.jnt_qposadr[trunk_jnt]
            data.qpos[q_adr : q_adr + 3] = [0.0, 0.0, 0.36]
            data.qpos[q_adr + 3 : q_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    # 2. 动态设置 12 关节标准站立角度
    for jname, q_val in zip(controller.joint_names, controller.stand_q):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            data.qpos[model.jnt_qposadr[jid]] = q_val

    # 3. 动态设置足球初始落点 (X = 1.2m, Z = 0.11m)
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")
    if ball_id != -1:
        ball_jnt = model.body_jntadr[ball_id]
        if ball_jnt != -1:
            b_adr = model.jnt_qposadr[ball_jnt]
            data.qpos[b_adr : b_adr + 3] = [1.2, 0.0, 0.11]
            data.qpos[b_adr + 3 : b_adr + 7] = [1.0, 0.0, 0.0, 0.0]

    mujoco.mj_forward(model, data)

def main():
    # 1. 解析模型路径并加载
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scene_path = os.path.join(current_dir, "field_scene.xml")

    if not os.path.exists(scene_path):
        raise FileNotFoundError(f"找不到场景文件: {scene_path}")

    print(f"正在加载场景: {scene_path} ...")
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    # 2. 初始化全能运动控制器
    controller = Go1MotionController(model, data)
    
    # 新增：初始化第三人称相机控制器（参数可按需调整）
    cam_controller = ThirdPersonCameraController(
        distance=1.8,
        elevation_deg=-20.0,
        lookat_height_offset=0.35,
        smoothness_azimuth=0.10,
        smoothness_elevation=0.10
    )

    # 3. 执行动态安全初值配置 (确保机器人稳固站立)
    reset_simulation_state(model, data, controller)

    # 动态获取刚体句柄
    robot_trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "soccer_ball")

    print("\n" + "="*60)
    print(" 🎮 Go1 足球机器人仿真控制台已就绪！")
    print(" ────────────────────────────────────────────────────────")
    print("  [W / S]   : 前进 / 后退")
    print("  [A / D]   : 原地左转 / 原地右转")
    print("  [Q / E]   : 侧向左平移 / 侧向右平移")
    print("  [X]       : 刹车急停 (速度立即归零)")
    print("  [空格键]  : 触发四足爆发跳跃动作")
    print(" ────────────────────────────────────────────────────────")
    print("  [1]       : 视角 1 - 机器人第三人称后上方追随视角")
    print("  [2]       : 视角 2 - 全场上帝俯瞰视角")
    print("  [3]       : 视角 3 - 正对球门特写视角")
    print("  [R]       : 重置场景回到初始站立")
    print("="*60 + "\n")

# 4. 启动分频解耦渲染主循环 (物理 500Hz + 渲染 60FPS)
    TARGET_RENDER_FPS = 60.0
    FRAME_INTERVAL = 1.0 / TARGET_RENDER_FPS
    # 每渲染一帧所需的物理子步进次数
    steps_per_frame = max(1, int(round(FRAME_INTERVAL / model.opt.timestep)))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_print_time = time.time()
        last_cmd_decay_time = time.time()

        while viewer.is_running():
            

            # 处理场景重置请求
            if keyboard_cmd["reset_requested"]:
                reset_simulation_state(model, data, controller)
                keyboard_cmd["vx"] = 0.0
                keyboard_cmd["vy"] = 0.0
                keyboard_cmd["wz"] = 0.0
                keyboard_cmd["reset_requested"] = False
                cam_controller.reset()

            # 处理跳跃触发
            if keyboard_cmd["jump_triggered"]:
                controller.trigger_jump()
                keyboard_cmd["jump_triggered"] = False

            # 高频物理动力学子步循环 (不阻塞渲染)
            for _ in range(steps_per_frame):
                controller.set_velocity_command(
                    keyboard_cmd["vx"],
                    keyboard_cmd["vy"],
                    keyboard_cmd["wz"]
                )
                controller.update()
                mujoco.mj_step(model, data)


            # 动态相机控制
            if keyboard_cmd["cam_mode"] == 1:
                #使用相机控制器获取滤波后的角度和注视点
                robot_pos = data.xpos[robot_trunk_id]
                robot_quat = data.xquat[robot_trunk_id]   # (w,x,y,z)
                az_deg, el_deg, lookat = cam_controller.update(robot_pos, robot_quat)
                
                # 设置 MuJoCo 查看器相机参数
                viewer.cam.lookat[:] = lookat
                viewer.cam.distance = cam_controller.distance   # 固定距离
                viewer.cam.azimuth = az_deg
                viewer.cam.elevation = el_deg
            elif keyboard_cmd["cam_mode"] == 2:
                #全场俯瞰
                viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
                viewer.cam.distance = 7.5
                viewer.cam.elevation = -45.0
                viewer.cam.azimuth = 90.0
            elif keyboard_cmd["cam_mode"] == 3:
                #球门特写
                viewer.cam.lookat[:] = [4.4, 0.0, 0.5]
                viewer.cam.distance = 3.5
                viewer.cam.elevation = -15.0
                viewer.cam.azimuth = 180.0

            # 按 60FPS 节拍同步一帧画面
            viewer.sync()

            # 键盘松开平滑减速
            current_time = time.time()
            if current_time - last_cmd_decay_time >= 0.15:
                keyboard_cmd["vx"] *= 0.88
                keyboard_cmd["vy"] *= 0.88
                keyboard_cmd["wz"] *= 0.88
                if abs(keyboard_cmd["vx"]) < 0.02: keyboard_cmd["vx"] = 0.0
                if abs(keyboard_cmd["vy"]) < 0.02: keyboard_cmd["vy"] = 0.0
                if abs(keyboard_cmd["wz"]) < 0.02: keyboard_cmd["wz"] = 0.0
                last_cmd_decay_time = current_time

            # 终端状态监控打印
            if current_time - last_print_time >= 0.5:
                ball_pos = data.xpos[ball_body_id]
                dist = np.linalg.norm(robot_pos[:2] - ball_pos[:2])
                print(f"[机身] 坐标: ({robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}) | "
                      f"指令: (vx={keyboard_cmd['vx']:.2f}, wz={keyboard_cmd['wz']:.2f}) | "
                      f"距球: {dist:.2f} m | 状态: {controller.jump_state}")
                last_print_time = current_time

if __name__ == "__main__":
    main()