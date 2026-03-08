import mujoco
import numpy as np


# NOTE: This is not a generic model manager
# Its tied to mujoco
# we are implicitly making an assumption on a 7dof robot (6 dof ur + 1 dof gripper)
# Same assumption is placed on IK, where we only compute ik for 6 dof (just the UR part)
class MujocoModelManager:
    def __init__(self, scene_path):
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.ee_site = "attachment_site"

        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.arm_dof_indices = [
            int(self.model.joint(name).dofadr) for name in self.arm_joint_names
        ]

    def get_joint_positions(self):
        return list(self.data.qpos[:7])

    def get_joint_velocities(self):
        return list(self.data.qvel[:7])

    def get_joint_accelerations(self):
        return list(self.data.qacc[:7])

    def fk(self, joint_angles):
        self.data.qpos[:6] = joint_angles[:6]
        mujoco.mj_kinematics(self.model, self.data)

        site_id = self.model.site(self.ee_site).id
        pos = self.data.site_xpos[site_id].copy()
        rot = self.data.site_xmat[site_id].reshape(3, 3).copy()

        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def ik(
        self, target_pos, target_rot=None, qinit=None, max_iter=200, tol=1e-4
    ) -> np.ndarray:
        q = np.array(qinit[:6], dtype=float) if qinit is not None else np.zeros(6)
        site_id = self.model.site(self.ee_site).id

        for _ in range(max_iter):
            self.data.qpos[:6] = q
            mujoco.mj_forward(self.model, self.data)

            # Position error
            pos_err = target_pos - self.data.site_xpos[site_id]

            # Orientation error
            if target_rot is not None:
                current_rot = self.data.site_xmat[site_id].reshape(3, 3)
                rot_err_mat = target_rot @ current_rot.T
                # Convert to axis-angle
                rot_err = (
                    np.array(
                        [
                            rot_err_mat[2, 1] - rot_err_mat[1, 2],
                            rot_err_mat[0, 2] - rot_err_mat[2, 0],
                            rot_err_mat[1, 0] - rot_err_mat[0, 1],
                        ]
                    )
                    * 0.5
                )
                err = np.concatenate([pos_err, rot_err])  # 6D
            else:
                err = pos_err  # 3D

            if np.linalg.norm(err) < tol:
                break

            # Full 6D Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

            lam = 0.01
            if target_rot is not None:
                J = np.vstack([jacp, jacr])[:, self.arm_dof_indices]  # 6x6
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)
            else:
                J = jacp[:, self.arm_dof_indices]  # 3x6
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

            q += dq

        return q
