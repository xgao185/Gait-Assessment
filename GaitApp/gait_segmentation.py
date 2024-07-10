import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def extract_gait_cycles(peaks, data, column_name):
    dataframes = {}
    for i in range(len(peaks) - 1):
        start = peaks[i]
        end = peaks[i + 1]
        segment = data[column_name][start:end]
        cycle_name = f"gaitcycle{i+1}"
        dataframes[cycle_name] = pd.DataFrame(segment)
    return dataframes


def interpolate_gait_cycles(dataframes, num_points=101):
    interpolated_dataframes = {}
    for cycle_name, df in dataframes.items():
        interpolated_segments = {}
        for column in df.columns:
            x = np.linspace(0, len(df) - 1, num=len(df))
            y = df[column].values
            x_new = np.linspace(0, len(df) - 1, num=num_points)
            y_interpolated = np.interp(x_new, x, y)
            interpolated_segments[column] = y_interpolated
        interpolated_dataframes[cycle_name] = pd.DataFrame(interpolated_segments)
    return interpolated_dataframes


def plot_interpolated_gait_cycles(interpolated_gait_cycles, joint_name, movement_name):
    plt.figure(figsize=(10, 6))
    for cycle_name, df in interpolated_gait_cycles.items():
        plt.plot(df.index, df.iloc[:, 0], label=cycle_name)  # 假设每个步态周期只有一列数据
    plt.xlabel('Time (%)')
    plt.ylabel(f'{joint_name} Angle (degrees)')
    plt.title(f'{joint_name} {movement_name} Angle')
    plt.legend()
    plt.show()
    st.pyplot(plt.gcf())  # 使用 Streamlit 显示图表


# 定义主函数
def display_gait_cycles_plots(joint_angles, right_peaks, left_peaks, position_data):
    # 创建三行，每行两列
    cols = [st.columns(2) for _ in range(3)]

    joints = ['Hip', 'Knee', 'Ankle']

    # 右侧髋、膝、踝
    for i, joint in enumerate(joints):
        with cols[i][1]:  # 放在右边一栏
            joint_name = f'Right {joint}'
            st.header(joint_name)
            segmentated_joint_angles = extract_gait_cycles(right_peaks, position_data, joint_angles[joint_name])
            interpolated_joint_angles = interpolate_gait_cycles(segmentated_joint_angles, 101)
            plot_interpolated_gait_cycles(interpolated_joint_angles, joint_name, 'Flexion/Extension')

    # 左侧髋、膝、踝
    for i, joint in enumerate(joints):
        with cols[i][0]:  # 放在左边一栏
            joint_name = f'Left {joint}'
            st.header(joint_name)
            segmentated_joint_angles = extract_gait_cycles(left_peaks, position_data, joint_angles[joint_name])
            interpolated_joint_angles = interpolate_gait_cycles(segmentated_joint_angles, 101)
            plot_interpolated_gait_cycles(interpolated_joint_angles, joint_name, 'Flexion/Extension')



