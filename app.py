import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import math

# --- 1. Helper Functions (Adapted for Streamlit) ---

def get_gradient_color(t):
    """Calculates a dynamic color based on time."""
    hue = (t * 120) % 360
    c = 1
    x = c * (1 - abs((hue / 60) % 2 - 1))
    
    # R, G, B values from 0 to 1
    if 0 <= hue < 60:
        r, g, b = c, x, 0
    elif 60 <= hue < 120:
        r, g, b = x, c, 0
    elif 120 <= hue < 180:
        r, g, b = 0, c, x
    elif 180 <= hue < 240:
        r, g, b = 0, x, c
    else: # 240 <= hue < 360
        r, g, b = x, 0, c if 240 <= hue < 300 else c
    
    # Convert to 0-255 range for OpenCV (BGR format is used for colors here)
    return (int(b * 255), int(g * 255), int(r * 255)) # BGR

def draw_smooth_line(img, pt1, pt2, color, thickness):
    """Draws a line with a faded glow effect."""
    cv2.line(img, pt1, pt2, color, thickness)
    cv2.line(img, pt1, pt2, tuple(c//2 for c in color), thickness+2)

def draw_smooth_circle(img, center, radius, color):
    """Draws a joint circle with a white core."""
    cv2.circle(img, center, radius, color, -1)
    cv2.circle(img, center, radius//2, (255, 255, 255), -1)

# Pose connections for the drawing
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                    (25, 27), (26, 28), (27, 31), (28, 32)]

# --- 2. Streamlit UI and Configuration ---

st.set_page_config(layout="wide")
st.title("🤖 CLONE TRACKER: MediaPipe Pose with Streamlit")

# Create Streamlit columns for UI organization
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Settings")
    # Streamlit sliders replace OpenCV Trackbars
    model_complexity = st.slider('Model Complexity (Requires Re-run)', 0, 2, 1, help="0 is fastest, 2 is highest accuracy. Requires re-running the script to take full effect.")
    min_det_conf = st.slider('Min Detection Confidence', 0.0, 1.0, 0.5, 0.05)
    min_track_conf = st.slider('Min Tracking Confidence', 0.0, 1.0, 0.5, 0.05)
    
    # Video and Status Placeholder
    st.subheader("Live Feed & Clone")
    stframe = st.empty()
    st.markdown("---")
    status_placeholder = st.empty()


# --- 3. MediaPipe and Main Loop Setup ---

# Initialize MediaPipe Pose with selected parameters
# Note: For Streamlit, Model Complexity is typically fixed after the first run
# unless the entire app is restarted (which is how Streamlit usually handles widget changes).
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, 
                     model_complexity=model_complexity, 
                     smooth_landmarks=True, 
                     min_detection_confidence=min_det_conf, 
                     min_tracking_confidence=min_track_conf)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = time.time()
frame_count = 0
trail_points = []
max_trail_length = 8
last_fps_update_time = time.time()
fps = 0

# Matplotlib 3D Setup inside Streamlit
with col2:
    st.header("3D Projection")
    # Create the Matplotlib figure for the 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('black')
    ax.set_title("3D Clone", color='cyan', fontsize=12)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Streamlit placeholder for the 3D plot
    st_plot = st.pyplot(fig)


# --- 4. Real-time Video Processing Loop ---
run = st.checkbox('Start Webcam', value=True)

while run:
    ret, frame = cap.read()
    if not ret:
        status_placeholder.error("Error: Could not read frame from webcam.")
        break
    
    frame = cv2.flip(frame, 1) # Mirror the image
    
    # Process Frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    current_time = time.time()
    wave_time = current_time - start_time
    
    # Dynamic Colors and Pulse Effect
    gradient_color = get_gradient_color(wave_time)
    pulse_intensity = 0.8 + 0.2 * math.sin(wave_time * 6)
    neon_color = tuple(int(c * pulse_intensity) for c in gradient_color)
    
    if results.pose_landmarks:
        height, width, _ = frame.shape
        
        # Calculate clone landmarks (mirrored coordinates)
        clone_landmarks_2d = [(int((1 - lm.x) * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]
        clone_landmarks_3d = [(1 - lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        
        glow_layer = np.zeros_like(frame, dtype=np.uint8)
        
        # Draw clone skeleton
        for i, j in POSE_CONNECTIONS:
            x1, y1 = clone_landmarks_2d[i]
            x2, y2 = clone_landmarks_2d[j]
            draw_smooth_line(glow_layer, (x1, y1), (x2, y2), neon_color, 6)
        
        clone_joint_points = []
        # Draw clone joints
        for x, y in clone_landmarks_2d:
            if 0 <= x < width and 0 <= y < height:
                clone_joint_points.append((x, y))
                draw_smooth_circle(glow_layer, (x, y), 8, neon_color)
        
        # Trail effects
        trail_points.append(clone_joint_points.copy())
        if len(trail_points) > max_trail_length:
            trail_points.pop(0)
        
        for t_idx, trail_frame in enumerate(trail_points[:-1]):
            trail_alpha = (t_idx / len(trail_points)) * 0.4
            trail_color = tuple(int(c * trail_alpha) for c in neon_color)
            for x, y in trail_frame:
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(glow_layer, (x, y), 3, trail_color, -1)
        
        # Blend layers
        frame = cv2.addWeighted(frame, 0.7, glow_layer, 0.8, 0)
        
        # Update 3D plot
        if frame_count % 3 == 0: # Update less frequently to save resources
            ax.clear()
            ax.set_facecolor('black')
            ax.set_title("3D Clone", color='cyan', fontsize=12)
            ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
            ax.grid(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            
            line_color = tuple(c/255 for c in neon_color)
            rgb_color = (line_color[2], line_color[1], line_color[0])
            
            for i, j in POSE_CONNECTIONS:
                ax.plot([lm[0] for lm in (clone_landmarks_3d[i], clone_landmarks_3d[j])], 
                        [-lm[1] for lm in (clone_landmarks_3d[i], clone_landmarks_3d[j])], 
                        [-lm[2] for lm in (clone_landmarks_3d[i], clone_landmarks_3d[j])], 
                        color=rgb_color, linewidth=3, alpha=0.8)
            
            for x, y, z in clone_landmarks_3d:
                ax.scatter(x, -y, -z, c=[rgb_color], s=50, alpha=0.8)
            
            # Re-render the Streamlit plot
            st_plot.pyplot(fig)
    
    else:
        # Scanning message
        height, width, _ = frame.shape
        search_text = "SCANNING..."
        text_x = (width - cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][0]) // 2
        text_y = (height + cv2.getTextSize(search_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0][1]) // 2
        scan_color = get_gradient_color(wave_time * 2)
        cv2.putText(frame, search_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, scan_color, 2, cv2.LINE_AA)
    
    # 5. FPS and Status Update
    frame_count += 1
    if current_time - last_fps_update_time >= 1:
        fps = frame_count / (current_time - last_fps_update_time)
        last_fps_update_time = current_time
        frame_count = 0
    
    fps_text = f"FPS: {fps:.1f}"
    conf_text = f"Det Conf: {min_det_conf:.2f} | Track Conf: {min_track_conf:.2f}"
    
    # Display overlay text on the frame
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Convert BGR frame back to RGB for Streamlit display
    frame_rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the live feed in Streamlit
    stframe.image(frame_rgb_display, channels="RGB", use_column_width=True)
    status_placeholder.info(f"Model Complexity: {model_complexity}")

# --- 5. Cleanup ---
cap.release()
