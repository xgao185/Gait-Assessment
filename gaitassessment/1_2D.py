import cv2
import mediapipe as mp
import streamlit as st
import tempfile
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from scipy.signal import find_peaks, butter, filtfilt

# Initializes MediaPipe's pose detection and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Streamlit interface. Sets the title and some custom CSS for the Streamlit app.
st.title("Human Motion Recognition System - Chendong\n\nAcknowledgements: UOA, FJTCM")
st.markdown("""
<style>
    .main-title {
        font-size: 24px !important;
    }
    .sub-title {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# Upload video file. Provides an interface for uploading a video file and displays a note about video length and positioning requirements.
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

st.markdown("""
 <div style="border:2px solid red; padding: 10px;">
     <p>由于内存限制，目前该程序仅接受10秒内的视频进行分析。由于数据分析结果是二维的，最理想的情况是让受试者正对摄像头进行冠状面运动，否则您可能需要对数据进行进一步分析。</p>
 </div>
 """, unsafe_allow_html=True)

# Load Model.  
@st.cache_resource
def load_model():
    return mp_pose.Pose()

pose = load_model()

@st.cache_data
def process_video(file_data):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file_data)
    tfile.close() # 关闭文件

    cap = cv2.VideoCapture(tfile.name)
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1  # Process every 1 frames
    landmarks = []
    frames = []
    ori_times = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Calculate video duration
        if frame_count == 1:
            video_duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
            if video_duration > 15:
                st.error("Video duration exceeds 10 seconds, please upload a shorter video.")
                cap.release()
                os.remove(tfile.name)
                return [], [], []

        if frame_count % frame_interval != 0:
            continue

        # Reduce resolution
        frame = cv2.resize(frame, (640, 480))

        # Convert the image from BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Perform pose detection, generate 3D position
        results = pose.process(image)

        if results.pose_landmarks:
            landmarks.append([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
            frames.append(frame_count)
            ori_times.append(frame_count / fps)

        # Force garbage collection
        gc.collect()

    cap.release()
    os.remove(tfile.name)
    return np.array(landmarks), frames, ori_times, fps, frame_interval

def plot_landmarks(landmarks, frame_idx, ax):
    ax.clear()
    connections = mp_pose.POSE_CONNECTIONS
    landmark_coords = landmarks[frame_idx]
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        start_coords = landmark_coords[start_idx]
        end_coords = landmark_coords[end_idx]
        ax.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], 'r-')
    ax.scatter(*zip(*[(lm[0], lm[1]) for lm in landmark_coords]), c='r', s=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

if uploaded_file is not None:
    landmarks, frames, ori_times, fps, frame_interval = process_video(uploaded_file.read())
    if len(landmarks) > 0:
        st.sidebar.title("Control Panel")
        frame_idx = st.sidebar.slider("Select Frame", 0, len(landmarks) - 1, 0)

        fig, ax = plt.subplots()
        plot_landmarks(landmarks, frame_idx, ax)
        st.pyplot(fig)
        left_foot_index_x = landmarks[:, 31, 0]
        
        # Apply low-pass filter to LeftFootIndex_x data
        fs = fps / frame_interval  # Sample rate (frames per second)
        cutoff = 2  # Desired cutoff frequency of the filter, Hz
        filtered_left_foot_index_x = butter_lowpass_filter(left_foot_index_x, cutoff, fs)

        mean_left_foot_index_x = np.mean(filtered_left_foot_index_x)
        
        # Find peaks in LeftFootIndex_x data
        peaks, _ = find_peaks(filtered_left_foot_index_x, height=mean_left_foot_index_x)
        
        # Find the middle frame index
        middle_frame = len(left_foot_index_x) // 2

       # Find the two peaks closest to the middle frame
        peak_distances = np.abs(peaks - middle_frame)
        closest_peaks_indices = peak_distances.argsort()[:2]
        closest_peaks = peaks[closest_peaks_indices]
        
        # Calculate the time difference between the two closest peaks
        time_difference = 60/(np.abs(closest_peaks[1] - closest_peaks[0]) / fps)
       
        # Create a DataFrame to display the time difference
        df_time_difference = pd.DataFrame({
           'Name': ['Time difference'],
           'Step frequency (steps/min)': [time_difference]
        })

        # Display the DataFrame in Streamlit
        st.write("Time Difference Table")
        st.dataframe(df_time_difference)
        
        # Convert frames to time
        closest_peaks_times = closest_peaks / fps
        times = np.array(frames) / fps
        
        # Plot LeftFootIndex_x data and peaks
        st.write("LeftFootIndex_x Peaks")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=times, y=left_foot_index_x, mode='lines', name='Original LeftFootIndex_x'))
        fig2.add_trace(go.Scatter(x=times, y=filtered_left_foot_index_x, mode='lines', name='Filtered LeftFootIndex_x'))
        fig2.add_trace(go.Scatter(x=closest_peaks_times, y=filtered_left_foot_index_x[closest_peaks], mode='markers', name='Closest Peaks', marker=dict(color='red', size=10)))
        
        # Add labels
        fig2.update_layout(
            xaxis_title='Time(s)',
            yaxis_title='Left Foot Position(X)'
        )
        
        st.plotly_chart(fig2)
        
        
        # Download button for landmarks data
        column_names = [
            "Nose_x", "Nose_y", "Nose_z",
            "LeftEyeInner_x", "LeftEyeInner_y", "LeftEyeInner_z",
            "LeftEye_x", "LeftEye_y", "LeftEye_z",
            "LeftEyeOuter_x", "LeftEyeOuter_y", "LeftEyeOuter_z",
            "RightEyeInner_x", "RightEyeInner_y", "RightEyeInner_z",
            "RightEye_x", "RightEye_y", "RightEye_z",
            "RightEyeOuter_x", "RightEyeOuter_y", "RightEyeOuter_z",
            "LeftEar_x", "LeftEar_y", "LeftEar_z",
            "RightEar_x", "RightEar_y", "RightEar_z",
            "MouthLeft_x", "MouthLeft_y", "MouthLeft_z",
            "MouthRight_x", "MouthRight_y", "MouthRight_z",
            "LeftShoulder_x", "LeftShoulder_y", "LeftShoulder_z",
            "RightShoulder_x", "RightShoulder_y", "RightShoulder_z",
            "LeftElbow_x", "LeftElbow_y", "LeftElbow_z",
            "RightElbow_x", "RightElbow_y", "RightElbow_z",
            "LeftWrist_x", "LeftWrist_y", "LeftWrist_z",
            "RightWrist_x", "RightWrist_y", "RightWrist_z",
            "LeftPinky_x", "LeftPinky_y", "LeftPinky_z",
            "RightPinky_x", "RightPinky_y", "RightPinky_z",
            "LeftIndex_x", "LeftIndex_y", "LeftIndex_z",
            "RightIndex_x", "RightIndex_y", "RightIndex_z",
            "LeftThumb_x", "LeftThumb_y", "LeftThumb_z",
            "RightThumb_x", "RightThumb_y", "RightThumb_z",
            "LeftHip_x", "LeftHip_y", "LeftHip_z",
            "RightHip_x", "RightHip_y", "RightHip_z",
            "LeftKnee_x", "LeftKnee_y", "LeftKnee_z",
            "RightKnee_x", "RightKnee_y", "RightKnee_z",
            "LeftAnkle_x", "LeftAnkle_y", "LeftAnkle_z",
            "RightAnkle_x", "RightAnkle_y", "RightAnkle_z",
            "LeftHeel_x", "LeftHeel_y", "LeftHeel_z",
            "RightHeel_x", "RightHeel_y", "RightHeel_z",
            "LeftFootIndex_x", "LeftFootIndex_y", "LeftFootIndex_z",
            "RightFootIndex_x", "RightFootIndex_y", "RightFootIndex_z"
        ]

        # Reshape landmarks and create DataFrame
        landmarks_df = pd.DataFrame(landmarks.reshape(-1, 33 * 3), columns=column_names)
        
        # Add frame number and time columns
        landmarks_df.insert(0, 'Frame', frames)
        landmarks_df.insert(0, 'Time', ori_times)
        
        csv = landmarks_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Landmarks Data as CSV",
            data=csv,
            file_name='landmarks.csv',
            mime='text/csv'
        )

        # Add contact information
        st.markdown("""
        <div style="border:2px solid red; padding: 10px;">
            <p>如果您是物理治疗师或从事动作评估相关职业，并对该程序有任何需求或反馈，请随时发送邮件至 chendongruan@gmail.com 或关注微信公众号“晨东康复说”留言。希望我们的合作能使这个程序帮助到更多人。</p>
        </div>
        """, unsafe_allow_html=True)