from abc import ABC, abstractmethod
from typing import Optional

from numpy.typing import NDArray


class RobotModelManagerBase(ABC):
    """
    Abstract base class for robot model managers.
    Simulator-agnostic.

    Subclasses must implement FK, IK, and joint state access methods.
    """

    def __init__(self):
        # Subclasses should define these:
        # - self.arm_joint_names: List[str]
        # - self.ee_name: str
        # - self.n_dof: int
        if not hasattr(self, "arm_joint_names") or not hasattr(self, "ee_name"):
            raise NotImplementedError(
                "Subclasses must define arm_joint_names and ee_name."
            )
        self.n_dof = len(self.arm_joint_names)

    # ---------------- Joint states ----------------
    @abstractmethod
    def get_joint_positions(self) -> NDArray:
        """Return current joint positions including gripper if present."""
        pass

    @abstractmethod
    def get_joint_velocities(self) -> NDArray:
        """Return current joint velocities including gripper if present."""
        pass

    @abstractmethod
    def get_joint_accelerations(self) -> NDArray:
        """Return current joint accelerations including gripper if present."""
        pass

    # ---------------- Forward kinematics ----------------
    @abstractmethod
    def fk(self, joint_angles: NDArray) -> NDArray:
        """
        Compute the 4x4 homogeneous transform of the end-effector
        given joint angles.
        Returns a 4x4 numpy array.
        """
        pass

    # ---------------- Inverse kinematics ----------------
    @abstractmethod
    def ik(
        self,
        target_pos: NDArray,
        target_rot: Optional[NDArray] = None,
        qinit: Optional[NDArray] = None,
        max_iter: int = 200,
        tol: float = 1e-4,
    ) -> NDArray:
        """
        Compute joint configuration that achieves target end-effector position
        and optional orientation.
        Returns joint angles as NDArray.
        """
        pass
