import mujoco
import numpy as np
import torch

class SensorAdapter:
    """
    传感器与空间运动学适配层：
    负责将 MuJoCo 底层物理量转化为符合强化学习策略规范的机身局部坐标系观测。
    """
    def __init__(self, model, data, trunk_body_name="trunk"):
        self.model = model
        self.data = data
        self.trunk_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, trunk_body_name)
        if self.trunk_body_id == -1:
            raise ValueError(f"无法在 MuJoCo 模型中找到躯干 Body: '{trunk_body_name}'")

    def get_projected_gravity(self):
        """
        计算重力向量在机身局部坐标系下的投影 (3维):
        利用 MuJoCo 内部维护的机身旋转矩阵 xmat (3x3 行优先) 转置投影 [0, 0, -1]。
        g_body = R^T * [0, 0, -1]^T = - [R_20, R_21, R_22]^T
        """
        xmat = self.data.xmat[self.trunk_body_id]
        # xmat[6], xmat[7], xmat[8] 为旋转矩阵第三行的三个分量
        gx = -xmat[6]
        gy = -xmat[7]
        gz = -xmat[8]
        return torch.tensor([gx, gy, gz], dtype=torch.float32)

    def get_body_angular_velocity(self):
        """
        计算在机身局部坐标系下的角速度 (3维):
        data.cvel 表达在世界对齐坐标系中，通过 R^T @ omega_world 逆变换为机身本体角速度。
        """
        cvel = self.data.cvel[self.trunk_body_id]
        omega_world = np.array(cvel[0:3], dtype=np.float32)

        xmat = self.data.xmat[self.trunk_body_id].reshape((3, 3))
        # R^T @ omega_world
        omega_body = xmat.T @ omega_world
        return torch.tensor(omega_body, dtype=torch.float32)

    def get_joint_states(self, qpos_indices, qvel_indices):
        """
        提取 12 关节的位置与速度张量
        """
        q = self.data.qpos[qpos_indices]
        dq = self.data.qvel[qvel_indices]
        return torch.tensor(q, dtype=torch.float32), torch.tensor(dq, dtype=torch.float32)

    def get_trunk_pose(self):
        """获取机身在世界系中的坐标与四元数 [w, x, y, z]"""
        pos = self.data.xpos[self.trunk_body_id].copy()
        quat = self.data.xquat[self.trunk_body_id].copy()
        return pos, quat