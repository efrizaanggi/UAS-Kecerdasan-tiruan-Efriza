from ultralytics import YOLO
import cv2
import streamlit as st
from PIL import Image
import numpy as np
from collections import Counter

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Process and display the detection results
def display_results(image, results, confidence_threshold=0.5):
    boxes = results.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    scores = results.boxes.conf.cpu().numpy()  # Confidence scores
    labels = results.boxes.cls.cpu().numpy()  # Class indices
    names = results.names  # Class names
    
    detected_objects = []
    
    for i in range(len(boxes)):
        if scores[i] > confidence_threshold:  # Confidence threshold
            x1, y1, x2, y2 = boxes[i].astype(int)
            label = names[int(labels[i])]
            score = scores[i]
            detected_objects.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image, detected_objects

# Main Streamlit app
def main():
    st.set_page_config(page_title="YOLO 11: Deteksi Objek", layout="wide", page_icon=":mag:")
    st.title("🔎 *YOLO 11*: Real-time Object Detection 🔍 Oleh: Efriza")
    st.sidebar.title("🔧 *Pengaturan*")
    
    model_path = "yolo11n.pt"  # Path to your YOLO model
    model = load_model(model_path)

    # Provide options: (Detection Camera, Upload Image)
    mode = st.sidebar.radio("Pilh Mode Mendeteksi", ("Real-Time Camera", "Unggah Gambar"))
    
    # Sidebar: Atur confidence threshold
    confidence_threshold = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, 0.5, 0.05)

    if mode == "Real-Time Camera":
        st.subheader("📹 *Mulai Mendeteksi*")
        run_detection = st.sidebar.checkbox("Mulai Deteksi", key="detection_control")
        # Open video capture if checkbox is active
        if run_detection:
            cap = cv2.VideoCapture(0) #Open Camera
            st_frame = st.empty()  # Placeholder for video frames
            st_detection_info = st.empty()  # Placeholder for detection information

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.warning("❌ Gagal dalam menangkap gambar dari kamera.")
                    break

                # Run YOLO detection
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
                results = model.predict(frame, imgsz=640)  # Perform detection
                
                # Draw results and collect detected objects
                frame, detected_objects = display_results(frame, results[0])
                
                # Display video feed
                st_frame.image(frame, channels="RGB", use_column_width=True)
                
                # Display detection information
                if detected_objects:
                    object_counts = Counter(detected_objects)
                    detection_info = "\n".join([f"{obj}: {count}" for obj, count in object_counts.items()])
                else:
                    detection_info = "⚪ Tidak ada objek yang terdeteksi."

                st_detection_info.text(detection_info)  # Update detection info text

                # Break the loop if checkbox is unchecked
                if not st.session_state.detection_control:
                    break
            
            cap.release()
            st.success("🌟 Deteksi objek dihentikan.")
    elif mode == "Unggah Gambar":
        st.subheader("📤 *Unggah Gambar untuk yang ingin dideteksi*")
        uploaded_file = st.file_uploader("Pilih file gambar:", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            image_np = np.array(image)

            # Run YOLO detection in images
            results = model.predict(image_np, imgsz=640)
            image_np, detected_objects = display_results(image_np, results[0], confidence_threshold)
            
            # Display image with detection result
            st.image(image_np, caption="Hasil Deteksi Objek Gambar", use_column_width=True)
            
            # Display detection info
            if detected_objects:
                object_counts = Counter(detected_objects)
                st.markdown("### *📊 Objek Terdeteksi:*")
                for obj, count in object_counts.items():
                    st.write(f"- *{obj}*: {count}")
            else:
                st.write("⚪ Tidak ada objek yang terdeteksi.")

if __name__ == "_main_":
    main()