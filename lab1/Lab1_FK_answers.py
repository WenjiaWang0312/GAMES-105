import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    # from IPython import embed
    # embed(header='34')
    with open(bvh_file_path, 'r') as f:
        content = f.readlines()
    joint_index = 0
    joint_name = ['RootJoint']
    joint_indent = [0]
    joint_parent = [-1]
    joint_offset = [np.zeros((1, 3))]
    for i in range(len(content)):
        if 'JOINT' in content[i] or 'End Site' in content[i]:
            joint_index += 1
            if 'JOINT' in content[i]:
                joint_name.append(content[i].split('JOINT')[1].replace(
                    '\n', '').replace(' ', ''))
                curr_indent = content[i].index('JOINT') / 4
            elif 'End Site' in content[i]:
                joint_name.append(joint_name[joint_index - 1] + '_end')
                curr_indent = content[i].index('End Site') / 4

            joint_indent.append(curr_indent)
            reversed_list = joint_indent[::-1]
            curr_parent = reversed_list.index(curr_indent - 1)
            curr_parent = len(joint_indent) - 1 - curr_parent
            joint_parent.append(curr_parent)
            curr_offset = np.array(content[i + 2].split('  ')[-3:]).astype(
                np.float32)
            joint_offset.append(curr_offset[None])
    # joint_name = None
    # joint_parent = None
    # joint_offset = None
    joint_offset = np.concatenate(joint_offset, 0)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset,
                             motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    translation = motion_data[frame_id, :3]
    joint_positions = np.zeros((len(joint_name), 3))
    joint_positions[0] = translation
    joint_orientations = np.zeros((len(joint_name), 3, 3))

    rotation_index = []
    for i, joint in enumerate(joint_name):
        if 'end' not in joint:
            rotation_index.append(i)
    rotation = motion_data[frame_id][3:].reshape(-1, 3)
    # rotation[3:] *=0
    rotation = R.from_euler('XYZ', rotation, degrees=True).as_matrix()
    joint_orientations[0] = rotation[0]
    for curr_i, parent_i in enumerate(joint_parent):
        if parent_i != -1:
            if curr_i in rotation_index:
                curr_i_rot = rotation_index.index(curr_i)
                joint_orientations[curr_i] = joint_orientations[
                    parent_i] @ rotation[curr_i_rot]
            else:
                joint_orientations[curr_i] = joint_orientations[parent_i]
            joint_positions[curr_i] = joint_positions[
                parent_i] + joint_orientations[parent_i] @ joint_offset[curr_i]

    joint_orientations = R.from_matrix(joint_orientations).as_quat()
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """

    joint_name1, joint_parent1, joint_offset1 = part1_calculate_T_pose(
        T_pose_bvh_path)
    joint_name2, joint_parent2, joint_offset2 = part1_calculate_T_pose(
        A_pose_bvh_path)
    parent1 = []
    for i in joint_parent1:
        parent1.append(joint_name1[i])
    parent2 = []
    for i in joint_parent2:
        parent2.append(joint_name2[i])

    d1 = dict(zip(joint_name1, parent1))
    d2 = dict(zip(joint_name2, parent2))
    d1['RootJoint'] = ''
    d2['RootJoint'] = ''
    assert d1 == d2

    def norm_vec(vec):
        return vec / np.linalg.norm(vec)

    def get_rotmat_pointy(vec):
        vec = norm_vec(vec)
        y = np.array([0, 1, 0])
        z = norm_vec(np.cross(vec, y))
        y = np.cross(vec, z)
        return np.array([vec, y, z])

    rotation_index1 = []
    for i, joint in enumerate(joint_name1):
        if 'end' not in joint:
            rotation_index1.append(i)

    rotation_index2 = []
    for i, joint in enumerate(joint_name2):
        if 'end' not in joint:
            rotation_index2.append(i)

    motion_data = load_motion_data(A_pose_bvh_path)

    new_motion_data = motion_data.copy()
    rot_offsets = []
    for i, j_name1 in enumerate(joint_name1):
        if joint_parent1[i] != -1:
            i2 = joint_name2.index(j_name1)
            if np.isclose(norm_vec(joint_offset1[i]),
                          norm_vec(joint_offset2[i2]),
                          atol=1e-2,
                          rtol=1e-2).all():
                rot_offset = np.eye(3)
            else:
                abs_rot_t = get_rotmat_pointy(joint_offset1[i])
                abs_rot_t_parent = get_rotmat_pointy(
                    joint_offset1[joint_parent1[i]])
                abs_rot_a = get_rotmat_pointy(joint_offset2[i2])
                abs_rot_a_parent = get_rotmat_pointy(
                    joint_offset2[joint_parent2[i2]])
                rel_rot_t = abs_rot_t_parent.T @ abs_rot_t
                rel_rot_a = abs_rot_a_parent.T @ abs_rot_a
                rot_offset = rel_rot_a.T @ rel_rot_t

            curr_i_rot1 = rotation_index1.index(joint_parent1[i])
            curr_i_rot2 = rotation_index2.index(joint_parent2[i2])
            num_frame = motion_data.shape[0]
            curr_joint_rotation = R.from_euler(
                'XYZ',
                motion_data[:, 3 + 3 * curr_i_rot2:6 + 3 * curr_i_rot2],
                degrees=True).as_matrix() @ rot_offset[None].repeat(
                    num_frame, 0)

            new_motion_data[:, 3 + 3 * curr_i_rot1:6 +
                            3 * curr_i_rot1] = R.from_matrix(
                                curr_joint_rotation).as_euler('XYZ',
                                                              degrees=True)
            rot_offsets.append(rot_offset)

    where_are_NaNs = np.isnan(new_motion_data)
    new_motion_data[where_are_NaNs] = 0

    return new_motion_data


def part3_retarget_func_easy(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name1, joint_parent1, joint_offset1 = part1_calculate_T_pose(
        T_pose_bvh_path)
    joint_name2, joint_parent2, joint_offset2 = part1_calculate_T_pose(
        A_pose_bvh_path)
    parent1 = []
    for i in joint_parent1:
        parent1.append(joint_name1[i])
    parent2 = []
    for i in joint_parent2:
        parent2.append(joint_name2[i])

    d1 = dict(zip(joint_name1, parent1))
    d2 = dict(zip(joint_name2, parent2))
    d1['RootJoint'] = ''
    d2['RootJoint'] = ''
    assert d1 == d2

    rotation_index1 = []
    for i, joint in enumerate(joint_name1):
        if 'end' not in joint:
            rotation_index1.append(i)

    rotation_index2 = []
    for i, joint in enumerate(joint_name2):
        if 'end' not in joint:
            rotation_index2.append(i)

    motion_data = load_motion_data(A_pose_bvh_path)

    new_motion_data = motion_data.copy()
    rot_offsets = []
    for i, j_name1 in enumerate(joint_name1):
        if joint_parent1[i] != -1 and 'end' not in j_name1:
            if 'Shoulder' in j_name1:
                if j_name1 == 'rShoulder':
                    angle = 45
                else:
                    angle = -45
            else:
                angle = 0
            rot_offset = R.from_euler('XYZ',
                                      np.array([0, 0, angle]),
                                      degrees=True).as_matrix()
            i2 = joint_name2.index(j_name1)

            curr_i_rot1 = rotation_index1.index(i)
            curr_i_rot2 = rotation_index2.index(i2)
            num_frame = motion_data.shape[0]
            curr_joint_rotation = R.from_euler(
                'XYZ',
                motion_data[:, 3 + 3 * curr_i_rot2:6 + 3 * curr_i_rot2],
                degrees=True).as_matrix() @ rot_offset[None].repeat(
                    num_frame, 0)

            new_motion_data[:, 3 + 3 * curr_i_rot1:6 +
                            3 * curr_i_rot1] = R.from_matrix(
                                curr_joint_rotation).as_euler('XYZ',
                                                              degrees=True)
            rot_offsets.append(rot_offset)

    where_are_NaNs = np.isnan(new_motion_data)
    new_motion_data[where_are_NaNs] = 0

    return new_motion_data
