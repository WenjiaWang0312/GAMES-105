import numpy as np
from scipy.spatial.transform import Rotation as R


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    def norm_vec(vec):
        return vec / np.linalg.norm(vec)

    def vec_length(vec):
        return np.linalg.norm(vec)

    def rodrigus_formula(vec1, vec2):
        """
        旋转向量转旋转矩阵
        """
        vec1 = norm_vec(vec1)
        vec2 = norm_vec(vec2)
        if np.allclose(vec1, vec2, atol=1e-3, rtol=1e-3):
            return np.eye(3)
        else:
            axis = np.cross(vec1, vec2)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(
                np.dot(vec1, vec2) /
                (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
            return R.from_rotvec(axis * angle).as_matrix()

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_orientations = R.from_quat(joint_orientations).as_matrix()

    num_iter = 20

    def get_all_child_joint(joint_parents, parent_index):
        all_child_joint = []
        for idx in range(len(joint_orientations)):
            curr_path = [idx]
            while joint_parents[curr_path[-1]] != -1:
                curr_path.append(joint_parents[curr_path[-1]])
            curr_path.remove(idx)
            if parent_index in curr_path:
                all_child_joint.append(idx)
        return all_child_joint

    def cyclic_coordinate_descent_ik(joint_positions, joint_orientations,
                                     joint_parent, path):
        """
        递归函数，计算逆运动学
        输入:
            joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
            kinematic_tree: 一个字典，key为关节名字，value为其父节点的名字
        输出:
            经过IK后的姿态
            joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        """
        kinematic_tree = joint_parent.copy()
        old_path = path.copy()
        path = path.copy()

        for i in range(len(path) - 1):
            kinematic_tree[path[i + 1]] = path[i]
        if 0 in path:
            path.remove(0)

        if 0 in old_path:
            for i, p_id in enumerate(kinematic_tree):
                if p_id == 0:
                    p_id = old_path[old_path.index(0) - 1]
                    kinematic_tree[i] = p_id
        kinematic_tree[path[0]] = -1

        for i in range(0, len(path) - 2):

            i = len(path) - 1 - i
            parent_index = path[i - 1]
            vec1 = joint_positions[path[-1]] - joint_positions[parent_index]
            vec2 = target_pose - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint = get_all_child_joint(kinematic_tree, parent_index)

            if (kinematic_tree[path[i]] != joint_parent[path[i]]):
                rot_joints = all_child_joint
                print(path[i], 'inverse')
            else:
                rot_joints = all_child_joint + [parent_index]
            joint_positions[all_child_joint] = (
                rot[None] @ (joint_positions[all_child_joint][:, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]

            # for eid in range(len(joint_orientations)):
            #     if eid not in kinematic_tree and eid in rot_joints:
            #         rot_joints.remove(eid)
            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]

        return joint_positions, joint_orientations

    def fabrik(joint_positions, joint_orientations, kinematic_tree, path,
               target_pose):
        """
        递归函数，计算逆运动学
        输入:
            joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
            kinematic_tree: 一个字典，key为关节名字，value为其父节点的名字
        输出:
            经过IK后的姿态
            joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
            joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        """
        offset = target_pose - joint_positions[path[-1]]
        joint_positions[path[-2:]] += offset
        joint_positions
        for i in range(len(path) - 1):
            # for i in [0, 1, 2, 3, 4]:

            i = len(path) - 1 - i
            parent_index = path[i - 1]
            vec1 = joint_positions[path[-1]] - joint_positions[parent_index]
            vec2 = target_pose - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint = get_all_child_joint(kinematic_tree, parent_index)

            rot_joints = all_child_joint + [parent_index]
            joint_positions[all_child_joint] = (
                rot[None] @ (joint_positions[all_child_joint][:, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]

            for eid in range(len(joint_orientations)):
                if eid not in kinematic_tree and eid in rot_joints:
                    rot_joints.remove(eid)
            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]
            # joint_orientations[rot_joints] = np.einsum(
            #     'ij, kmi->kmj', rot, joint_orientations[rot_joints])
        return joint_positions, joint_orientations

    for iter_idx in range(num_iter):
        thershold = 0.01

        whole_length = 0
        for i in range(len(path) - 1):
            whole_length += vec_length(joint_positions[path[i + 1]] -
                                       joint_positions[path[i]])
        target_length = vec_length(target_pose - joint_positions[path[0]])
        if target_length > whole_length:
            thershold = target_length - whole_length

        if not vec_length(joint_positions[path[-1]] - target_pose) < thershold:
            joint_positions, joint_orientations = cyclic_coordinate_descent_ik(
                joint_positions, joint_orientations, meta_data.joint_parent,
                path, meta_data.joint_name)
            # joint_positions, joint_orientations = fabrik(
            #     joint_positions, joint_orientations, meta_data.joint_parent,
            #     path, target_pose)
            print(joint_positions[path[-1]] - target_pose, iter_idx)
    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations


def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """

    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations,
                             left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations