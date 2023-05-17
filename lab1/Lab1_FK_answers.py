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
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
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
                joint_name.append(content[i].split('JOINT')[1].replace('\n', '').replace(' ', ''))
                curr_indent = content[i].index('JOINT')/4
            elif 'End Site' in content[i]:
                joint_name.append(joint_name[joint_index-1] + '_end')
                curr_indent = content[i].index('End Site')/4

            joint_indent.append(curr_indent)
            reversed_list = joint_indent[::-1]
            curr_parent = reversed_list.index(curr_indent-1)
            curr_parent = len(joint_indent) - 1 - curr_parent
            joint_parent.append(curr_parent)
            curr_offset = np.array(content[i+2].split('  ')[-3:]).astype(np.float32)
            joint_offset.append(curr_offset[None])
    # joint_name = None
    # joint_parent = None
    # joint_offset = None
    joint_offset = np.concatenate(joint_offset, 0)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
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
    joint_positions = None
    joint_orientations = None
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
    motion_data = None
    return motion_data
