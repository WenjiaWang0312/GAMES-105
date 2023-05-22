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

    num_iter = 100
    origin_pos = joint_positions.copy()
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
                                     joint_parent, path, target_pose, out_of_stretch):
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

        if out_of_stretch:
            start = len(path) - 3
        else:
            start = 0
        for i in range(start, len(path) - 2):

            curr_index = path[len(path) - 1 - i]
            parent_index = path[curr_index - 1]
            vec1 = joint_positions[path[-1]] - joint_positions[parent_index]
            vec2 = target_pose - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint = get_all_child_joint(kinematic_tree, parent_index)

            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = all_child_joint
                print(curr_index, 'inverse')
            else:
                rot_joints = all_child_joint + [parent_index]
            joint_positions[all_child_joint] = (
                rot[None] @ (joint_positions[all_child_joint][:, :, None] -
                            joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]

            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]

        return joint_positions, joint_orientations

    def fabrik(joint_positions, joint_orientations, joint_parent, path,
               target_pose, out_of_stretch, origin_pos):
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

        protected_ids = []
        if 0 in old_path:
            for i, p_id in enumerate(kinematic_tree):
                if p_id == 0:
                    new_p_id = old_path[old_path.index(0) - 1]
                    kinematic_tree[i] = new_p_id
                    protected_ids.append(i)
        kinematic_tree[path[0]] = -1
            
        # from IPython import embed
        # embed()
        if kinematic_tree[0] in protected_ids:
            protected_ids.remove(kinematic_tree[0])

        root = origin_pos[path[0]].copy()

        last_position_state = joint_positions.copy()
        # stage1
        for i in range(len(path)-1):
        
            curr_index = path[len(path) - 1 - i]

            if i == 0:
                joint_positions[curr_index] = target_pose

            last_end = joint_positions[curr_index]
            parent_index = path[len(path) - 2 - i]
            vec1 = last_position_state[curr_index] - joint_positions[parent_index]
            vec2 = last_end - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint_start = get_all_child_joint(kinematic_tree, parent_index)
            all_child_joint_end = get_all_child_joint(kinematic_tree, curr_index)
            curr_bone_child = list(set(all_child_joint_start) - set(all_child_joint_end) - set([curr_index]))

            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = curr_bone_child + [curr_index]
                # print(curr_index, 'inverse')
            else:
                rot_joints = curr_bone_child + [parent_index]

            if 0 in old_path:
                if curr_index in protected_ids and curr_index in path and len(path) - 1 - i >= old_path.index(0):
                    rot_joints = curr_bone_child
            pos_joints = curr_bone_child + [parent_index]
            # rot_joints = curr_bone_child + [parent_index]
            # from IPython import embed
            # embed()
            curr_position_rotated = (
                rot[None] @ (last_position_state[curr_index][None, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]
            bone_length = vec_length(origin_pos[curr_index] - origin_pos[parent_index])
            joint_positions[pos_joints] = (
                rot[None] @ (joint_positions[pos_joints][:, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0] + norm_vec(vec2) * (vec_length(vec2) - bone_length)

            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]
            

        last_position_state = joint_positions.copy()

        for i in range(len(path)-1):

            curr_index = path[i]

            if i == 0:
                joint_positions[curr_index] = root

            last_end = joint_positions[curr_index]
            parent_index = path[i+1]
            vec1 = last_position_state[curr_index] - joint_positions[parent_index]
            vec2 = last_end - joint_positions[parent_index]
            rot = rodrigus_formula(vec1, vec2)
            all_child_joint_start = get_all_child_joint(kinematic_tree, curr_index)
            all_child_joint_end = get_all_child_joint(kinematic_tree, parent_index)
            curr_bone_child = list(set(all_child_joint_start) - set(all_child_joint_end) - set([parent_index]))

            # from IPython import embed
            # embed()
            if (kinematic_tree[curr_index] != joint_parent[curr_index]):
                rot_joints = curr_bone_child + [parent_index]
                # print(curr_index, 'inverse')
            else:
                rot_joints = curr_bone_child + [curr_index]

            if 0 in old_path:
                if curr_index in protected_ids and curr_index in path and len(path) - 1 - i >= old_path.index(0):
                    rot_joints = curr_bone_child
            pos_joints = curr_bone_child + [parent_index]
 
            curr_position_rotated = (
                rot[None] @ (last_position_state[curr_index][None, :, None] -
                             joint_positions[parent_index][None, :, None]) +
                joint_positions[parent_index][None, :, None])[..., 0]
            bone_length = vec_length(origin_pos[curr_index] - origin_pos[parent_index])
            joint_positions[pos_joints] = (
                rot[None] @ (joint_positions[pos_joints][:, :, None] -
                             joint_positions[parent_index][None, :, None]) + 
                joint_positions[parent_index][None, :, None])[..., 0] +  norm_vec(vec2) * (vec_length(vec2) - bone_length)

            joint_orientations[
                rot_joints] = rot[None] @ joint_orientations[rot_joints]
            
        return joint_positions, joint_orientations

    whole_length = 0
    for i in range(len(path) - 1):
        whole_length += vec_length(meta_data.joint_initial_position[path[i + 1]] -
                                    meta_data.joint_initial_position[path[i]])
        
    for iter_idx in range(num_iter):
        thershold = 0.01
        out_of_stretch = False


        target_length = vec_length(target_pose - origin_pos[path[0]])
        if target_length > whole_length:
            out_of_stretch = True
            thershold = target_length - whole_length + 0.01 * target_length / whole_length
        
        if not vec_length(joint_positions[path[-1]] - target_pose) < thershold:
            # joint_positions, joint_orientations = cyclic_coordinate_descent_ik(
            #     joint_positions, joint_orientations, meta_data.joint_parent,
            #     path, target_pose, out_of_stretch)
            joint_positions, joint_orientations = fabrik(
                joint_positions, joint_orientations, meta_data.joint_parent,
                path, target_pose, out_of_stretch, meta_data.joint_initial_position)
            # print(joint_positions[path[-1]] - target_pose, iter_idx)
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