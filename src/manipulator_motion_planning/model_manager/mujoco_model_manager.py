from typing import List

import mujoco
import numpy as np
from manipulator_motion_planning.model_manager.base import RobotModelManagerBase
from numpy.typing import NDArray


class MujocoModelManager(RobotModelManagerBase):
    def __init__(self, scene_path: str):
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
        self.n_dof = len(self.arm_joint_names)
        self.arm_dof_indices = [
            int(self.model.joint(name).dofadr) for name in self.arm_joint_names
        ]

    def get_joint_positions(self) -> List:
        return list(self.data.qpos[:7])

    def get_joint_velocities(self) -> List:
        return list(self.data.qvel[:7])

    def get_joint_accelerations(self) -> List:
        return list(self.data.qacc[:7])

    def fk(self, joint_angles: NDArray) -> NDArray:
        """Performs forward kinematics to compute
        the transformation matrix corresponding to the
        desired joint angles"""
        # Copy desired joint angles into model's qpos array
        self.data.qpos[:6] = joint_angles[:6]
        # Set the qpos to the model
        mujoco.mj_kinematics(self.model, self.data)

        # Get end-effector site id
        site_id = self.model.site(self.ee_site).id
        # Get pose and orient of end effector
        pos = self.data.site_xpos[site_id].copy()
        rot = self.data.site_xmat[site_id].reshape(3, 3).copy()

        # Set and return transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T

    def ik(
        self,
        target_pos: NDArray,
        target_rot: NDArray = None,
        qinit: NDArray = None,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> NDArray:
        """Performs inverse kinematics to compute the joint configuration
        that achieves the target end-effector pos and orientation"""
        # Get first joint angles: either provided guess (qinit) or zeros
        q = np.array(qinit[:6], dtype=float) if qinit is not None else np.zeros(6)
        # Get end-effector site id
        site_id = self.model.site(self.ee_site).id

        # Iteratively solve for target qpos
        # until less than tol error
        for _ in range(max_iter):
            self.data.qpos[:6] = q
            # Advance simulation using data provided
            mujoco.mj_forward(self.model, self.data)

            # Position error
            pos_err = target_pos - self.data.site_xpos[site_id]

            # Orientation error
            if target_rot is not None:
                current_rot = self.data.site_xmat[site_id].reshape(3, 3)
                # Rotation needed = target * inverse(current)
                # Digression:
                # inverse == transpose for orthogonal matrices like rotations.
                # An orthogonal matrix is one where the columns are orthonormal
                # i.e. mat * mat_T = mat_T * mat = I
                # Orthogonal matrices could also be reflections, so rotations are
                # a subset called SO(3) (Special Orthogonal 3) of the lie group.
                # A Lie group is a smooth group that looks like a curve or a smooth
                # surface that is differentiable (possibly in n dimensions).
                rot_err_mat = target_rot @ current_rot.T
                # Convert to axis-angle using skew-symmetric extraction of a rotation matrix
                # For a rotation error matrix R, the angular error vector ω can be approximated as:
                #
                #        1
                # ω ≈  ----- * [ R32 - R23
                #        2       R13 - R31
                #                R21 - R12 ]
                #
                # For small rotations this vector approximates the axis-angle rotation error.
                # Since axis-angle is a rotation angle (θ) and a direction vector (u), and this
                # trick assumes sin(θ) ≈ θ, which is true for small angles.
                # In robotics this is often called the orientation residual in so(3),
                # where so(3) is the Lie algebra corresponding to the rotation group SO(3).
                #
                # A skew-symmetric matrix is a square matrix that equals the negative of its transpose.
                # Key characteristics include zero-value main diagonal elements and off-diagonal elements
                # that are opposites. These matrices have zero trace, purely imaginary or zero eigenvalues,
                # and any square matrix can be expressed as a sum of symmetric and skew-symmetric components.
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

            # If combined error of all axes < tol
            if np.linalg.norm(err) < tol:
                break

            # Full 6D Jacobian
            # nv - number of generalized velocity coordinates (DoFs - Degrees of Freedom).
            # nu - total number of control inputs (actuators) defined in your MJCF model.
            # jacp Jp​ = ∂x / ∂q​ : Linear Jacobian (3 x nv).
            #        Maps joint velocities to the 3D translational velocity of the site.
            #        v_site = jacp @ qdot
            # jacr Jr​ = ∂θ / ∂q : Rotational Jacobian (3 x nv).
            #        Maps joint velocities to the 3D angular velocity of the site.
            #        omega_site = jacr @ qdot
            # where:
            # qdot        = joint velocity vector
            # v_site      = linear velocity of the site (x, y, z)
            # omega_site  = angular velocity of the site (wx, wy, wz)
            # nv          = number of velocity DoFs in the model
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

            # Damping term for damped least-squares IK (prevents instability near singularities)
            # Also known as Levenberg-Marquardt IK.
            lam = 0.01

            if target_rot is not None:
                # Stack linear and rotational Jacobians -> full 6D Jacobian
                # maps joint velocities -> [linear_vel, angular_vel]
                J = np.vstack([jacp, jacr])[:, self.arm_dof_indices]  # shape: 6 x 6

                # Damped pseudoinverse solve:
                # dq = J^T (J J^T + λI)^(-1) err
                # Regular pseudo-inverse -> J^T (J J^T)^(-1)
                # Damping term -> λI
                # Computes joint update that best reduces 6D pose error
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), err)

            else:
                # Position-only IK (ignore orientation)
                J = jacp[:, self.arm_dof_indices]  # shape: 3 x 6

                # Same damped least-squares solve but for position error only
                dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

            # Add the delta joint-angles to current joint angles
            q += dq

        return q
