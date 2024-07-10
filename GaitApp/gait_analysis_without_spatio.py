# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:20:40 2024

@author: Xi Gao and Xingye
"""
import os
# 安装缺少的libGL库
os.system('apt-get update && apt-get install -y libgl1-mesa-glx')

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime
import tempfile

from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from gait_cycle_segmentation import find_peaks_and_valleys
from gait_segmentation import display_gait_cycles_plots
# from spatiotemporal_parameters import stance_phase, velocity, LRcadence, step_length
# from calibration import get_calibration_ratio

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_angle_between_vectors(p1, p2, p3, p4):
    """
    Calculate the angle between two vectors defined by four points in 2D space.

    Args:
    p1, p2: The points defining the first vector (p1 -> p2).
    p3, p4: The points defining the second vector (p3 -> p4).

    Returns:
    angle: The angle between the two vectors in degrees.
    """
    # Convert points to numpy arrays
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = np.array(p3)
    p4 = np.array(p4)

    # Calculate vectors
    v1 = p2 - p1
    v2 = p4 - p3

    # Calculate the dot product and the magnitudes of the vectors
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_angle = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure the cosine value is within the valid range [-1, 1] to avoid numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def process_video(video_source, is_file=False):
    if is_file:
        cap = cv2.VideoCapture(video_source)
    else:
        cap = cv2.VideoCapture(0)  # Use webcam

    frame_count = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Ensure video starts from the first frame

    while cap.isOpened() and st.session_state.exercise_active:
        ret, frame = cap.read()
        if not ret:
            break

        # # Ensure video time starts from the beginning
        # video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            landmark_dict = {}
            for lm in mp_pose.PoseLandmark:
                landmark_dict[f"{lm.name}_x"] = landmarks[lm.value].x
                landmark_dict[f"{lm.name}_y"] = landmarks[lm.value].y

            # Get coordinates
            Rhip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            Rknee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            Rankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            Rheel = [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            Rtoe = [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                    landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y]
            Rhip2 = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x + 1,
                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            Lhip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            Lknee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            Lankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            Lheel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
            Ltoe = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
            Lhip2 = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x + 1,
                     landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate knee angle
            R_knee_angle: float = 180 - calculate_angle(Rhip, Rknee, Rankle)
            L_knee_angle = 180 - calculate_angle(Lhip, Lknee, Lankle)
            R_ankle_angle = 90 - calculate_angle_between_vectors(Rankle, Rknee, Rheel, Rtoe)
            L_ankle_angle = 90 - calculate_angle_between_vectors(Lankle, Lknee, Lheel, Ltoe)
            R_hip_angle = 90 - calculate_angle(Rhip2, Rhip, Rknee)
            L_hip_angle = 90 - calculate_angle(Lhip2, Lhip, Lknee)

            # Save position data
            st.session_state.position_data.append({
                'timestamp': datetime.datetime.now(),
                'video_time': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000,  # Convert milliseconds to seconds
                'frame': frame_count,
                'L_knee_angle': L_knee_angle,
                'R_knee_angle': R_knee_angle,
                'L_ankle_angle': L_ankle_angle,
                'R_ankle_angle': R_ankle_angle,
                'L_hip_angle': L_hip_angle,
                'R_hip_angle': R_hip_angle,
                **landmark_dict
            })

        except Exception as e:
            print(f"Error extracting landmarks: {e}")

        frame_count += 1

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Display the resulting frame
        st.session_state.video_frame.image(image, channels="BGR")

    cap.release()


def main():
    st.title("Video Pose Detection")

    # Initialize session_state variables
    if 'exercise_active' not in st.session_state:
        st.session_state.exercise_active = False
    if 'position_data' not in st.session_state:
        st.session_state.position_data = []
    if 'video_frame' not in st.session_state:
        st.session_state.video_frame = st.empty()
    if 'chart_placeholder' not in st.session_state:
        st.session_state.chart_placeholder = st.empty()
    if 'segmentation_done' not in st.session_state:
        st.session_state.segmentation_done = False
    if 'left_strike_peaks' not in st.session_state:
        st.session_state.left_strike_peaks = []
    if 'right_strike_peaks' not in st.session_state:
        st.session_state.right_strike_peaks = []
    if 'left_toeoff_peaks' not in st.session_state:
        st.session_state.left_toeoff_peaks = []
    if 'right_toeoff_peaks' not in st.session_state:
        st.session_state.right_toeoff_peaks = []
    if 'left_toe_hip_distance' not in st.session_state:
        st.session_state.left_toe_hip_distance = []
    if 'right_toe_hip_distance' not in st.session_state:
        st.session_state.right_toe_hip_distance = []
    if 'left_heel_hip_distance' not in st.session_state:
        st.session_state.left_heel_hip_distance = []
    if 'right_heel_hip_distance' not in st.session_state:
        st.session_state.right_heel_hip_distance = []

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        if st.button("Start Estimation"):
            # Initialize session state variables for new video
            st.session_state.position_data = []
            st.session_state.exercise_active = True
            process_video(tfile.name, is_file=True)

        if st.button("Stop Estimation"):
            st.session_state.exercise_active = False
            # st.session_state.csv_ready = True

        if st.button("Save Data"):
            df = pd.DataFrame(st.session_state.position_data)

            # Apply filter to the position data
            fs = 30.0  # Sampling frequency (assumed fps)
            cutoff = 6.0  # Cutoff frequency

            filtered_columns = ['L_knee_angle', 'R_knee_angle', 'L_ankle_angle', 'R_ankle_angle', 'L_hip_angle',
                                'R_hip_angle'] + [f"{lm.name}_x" for lm in mp_pose.PoseLandmark] + [f"{lm.name}_y" for
                                                                                                    lm in
                                                                                                    mp_pose.PoseLandmark]
            for col in filtered_columns:
                df[f'{col}_filtered'] = butter_lowpass_filter(df[col], cutoff, fs)
            st.session_state.position_data = df
            # Create a line chart for filtered angles
            chart_data = pd.DataFrame({
                'RHip Angle': df['R_hip_angle_filtered'],
                'RKnee Angle': df['R_knee_angle_filtered'],
                'RAnkle Angle': df['R_ankle_angle_filtered']
            })
            st.session_state.chart_placeholder.line_chart(chart_data)

            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download position data as CSV",
                    data=csv,
                    file_name='position_data.csv',
                    mime='text/csv'
                )
                st.success("CSV file is ready for download")
            else:
                st.warning("No data to save.")

        if st.button("Segmentation"):
            if st.session_state.position_data.empty:
                st.warning("Please save data before performing segmentation.")

            else:
                df = pd.DataFrame(st.session_state.position_data)
                left_hip = df['LEFT_HIP_x_filtered']
                left_heel = df['LEFT_HEEL_x_filtered']
                left_foot_index = df['LEFT_FOOT_INDEX_x_filtered']
                right_hip = df['RIGHT_HIP_x_filtered']
                right_heel = df['RIGHT_HEEL_x_filtered']
                right_foot_index = df['RIGHT_FOOT_INDEX_x_filtered']

                # use heel minus hip to find the heel strike
                left_heel_hip_distance = left_heel - left_hip
                right_heel_hip_distance = right_heel - right_hip
                left_strike_peaks = find_peaks_and_valleys(left_heel_hip_distance)
                right_strike_peaks = find_peaks_and_valleys(right_heel_hip_distance)

                # use toe minus hip to find the toe off
                left_toe_hip_distance = left_foot_index - left_hip
                right_toe_hip_distance = right_foot_index - right_hip
                left_toeoff_peaks = find_peaks_and_valleys(-left_toe_hip_distance)
                right_toeoff_peaks = find_peaks_and_valleys(-right_toe_hip_distance)

                # Lstance_phase = stance_phase(np.array(left_toeoff_peaks), np.array(left_strike_peaks))
                # Rstance_phase = stance_phase(np.array(right_toeoff_peaks), np.array(right_strike_peaks))
                # Lcadence, Rcadence = LRcadence(left_strike_peaks, right_strike_peaks, 30)
                #
                # scale_factor, image_width = get_calibration_ratio('Calibration', 2.5)
                # left_step_pixel = step_length(left_heel, right_heel, left_strike_peaks, image_width)
                # Lstep_length = left_step_pixel * scale_factor
                # right_step_pixel = step_length(right_heel, left_heel, right_strike_peaks, image_width)
                #
                # Rstep_length = right_step_pixel * scale_factor
                #
                # Lvelocity = velocity(Lstep_length, Lcadence)
                # Rvelocity = velocity(Rstep_length, Rcadence)
                #
                # stance_phase_L_output: str = f"{np.mean(Lstance_phase):.2f}±{np.std(Lstance_phase):.2f}"
                # stance_phase_R_output = f"{np.mean(Rstance_phase):.2f}±{np.std(Rstance_phase):.2f}"
                # step_length_L_output = f"{np.mean(Lstep_length):.2f}±{np.std(Lstep_length):.2f}"
                # step_length_R_output = f"{np.mean(Rstep_length):.2f}±{np.std(Rstep_length):.2f}"
                # cadence_L_output = f"{np.mean(Lcadence):.2f}±{np.std(Lcadence):.2f}"
                # cadence_R_output = f"{np.mean(Rcadence):.2f}±{np.std(Rcadence):.2f}"
                # Lvelocity_output = f"{Lvelocity:.2f}"
                # Rvelocity_output = f"{Rvelocity:.2f}"
                #
                # # 创建表格数据
                # data = {
                #     'Side': ['left', 'right'],
                #     'Velocity (m/s)': [Lvelocity_output, Rvelocity_output],
                #     'Cadence (steps/min)': [cadence_L_output, cadence_R_output],
                #     'Step Length (m)': [step_length_L_output, step_length_R_output],
                #     'Stance Phase (%)': [stance_phase_L_output, stance_phase_R_output]
                # }
                #
                # # 创建并显示表格
                # st.table(data)

                st.session_state.left_strike_peaks = left_strike_peaks
                st.session_state.right_strike_peaks = right_strike_peaks

                st.session_state.left_toeoff_peaks = left_toeoff_peaks
                st.session_state.right_toeoff_peaks = right_toeoff_peaks

                st.session_state.left_heel_hip_distance = left_heel_hip_distance
                st.session_state.right_heel_hip_distance = right_heel_hip_distance
                st.session_state.left_toe_hip_distance = left_toe_hip_distance
                st.session_state.right_toe_hip_distance = right_toe_hip_distance
                st.session_state.segmentation_done = True

    '''
    plot find peaks result

    '''

    if st.session_state.segmentation_done:
        st.write("Segmentation done:", st.session_state.segmentation_done)

        # 创建两列布局
        col1, col2 = st.columns(2)
        with col1:
            # 绘制左腿图表
            fig_left, ax_left = plt.subplots(figsize=(5, 3))
            ax_left.plot(st.session_state.left_heel_hip_distance, label='Left Heel-Hip Distance')
            ax_left.plot(st.session_state.left_strike_peaks,
                         st.session_state.left_heel_hip_distance[st.session_state.left_strike_peaks], "x",
                         label='Peaks')
            print(type(st.session_state.left_strike_peaks))
            print(type(st.session_state.left_heel_hip_distance))
            # ax_left.plot(st.session_state.left__valleys, st.session_state.left_distance[st.session_state.left_valleys],
            #              "o",
            #              label='Valleys')
            ax_left.legend()
            ax_left.set_title('Left Heel-Hip Peaks')
            st.pyplot(fig_left)
        with col2:
            # 绘制右腿图表
            fig_right, ax_right = plt.subplots(figsize=(5, 3))
            ax_right.plot(st.session_state.right_heel_hip_distance, label='Right Heel-Hip Distance')
            ax_right.plot(st.session_state.right_strike_peaks,
                          st.session_state.right_heel_hip_distance[st.session_state.right_strike_peaks],
                          "x",
                          label='Peaks')
            # ax_right.plot(st.session_state.right_valleys,
            #               st.session_state.right_distance[st.session_state.right_valleys],
            #               "o",
            #               label='Valleys')
            ax_right.legend()
            ax_right.set_title('Right Heel-Hip Peaks')
            st.pyplot(fig_right)

        col3, col4 = st.columns(2)
        with col3:
            # 绘制左腿额外图表
            fig_left_extra, ax_left_extra = plt.subplots(figsize=(5, 3))
            ax_left_extra.plot(st.session_state.left_toe_hip_distance, label='Left Toe-Knee Distance')
            ax_left_extra.plot(st.session_state.left_toeoff_peaks,
                               st.session_state.left_toe_hip_distance[st.session_state.left_toeoff_peaks], "x",
                               label='Left Toe Off Peaks')
            ax_left_extra.legend()
            ax_left_extra.set_title('Left Toe Off Peaks')
            st.pyplot(fig_left_extra)
        with col4:
            # 绘制右腿额外图表
            fig_right_extra, ax_right_extra = plt.subplots(figsize=(5, 3))
            ax_right_extra.plot(st.session_state.right_toe_hip_distance, label='Right Toe Hip Distance')
            ax_right_extra.plot(st.session_state.right_toeoff_peaks,
                                st.session_state.right_toe_hip_distance[st.session_state.right_toeoff_peaks], "x",
                                label='Right Toe Off Peaks')
            ax_right_extra.legend()
            ax_right_extra.set_title('Right Toe Off Peaks')
            st.pyplot(fig_right_extra)

        joint_angles = {'Right Hip': 'R_hip_angle_filtered',
                        'Right Knee': 'R_knee_angle_filtered',
                        'Right Ankle': 'R_ankle_angle_filtered',
                        'Left Hip': 'L_hip_angle_filtered',
                        'Left Knee': 'L_knee_angle_filtered',
                        'Left Ankle': 'L_ankle_angle_filtered'}

        display_gait_cycles_plots(joint_angles, st.session_state.right_strike_peaks, st.session_state.left_strike_peaks,
                                  st.session_state.position_data)


if __name__ == "__main__":
    main()
