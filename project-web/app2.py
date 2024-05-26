import cv2
import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

def draw_boxes(image, results):
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            cords = box.xyxy[0].tolist()
            cords = [round(x) for x in cords]
            conf = round(box.conf[0].item(), 2)
            # Draw bounding box
            cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), (0, 255, 0), 2)
            # Put label and confidence
            text = f"{class_id}: {conf}"
            cv2.putText(image, text, (cords[0], cords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image

def app():
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ['Single class OofS', 'Single class misplaced', 'Multiclass', 'Extension Model'])

    if selection == 'Single class OofS':
        custom_model_path = "C:/Users/bheja/OneDrive/Desktop/best_models/outofstock.pt"
    elif selection == 'Single class misplaced':
        custom_model_path = "C:/Users/bheja/OneDrive/Desktop/best_models/misplaced.pt"
    elif selection == 'Multiclass':
        custom_model_path = "C:/Users/bheja/OneDrive/Desktop/best_models/multiclass.pt"
    elif selection == 'Extension Model':
        custom_model_path = "C:/Users/bheja/OneDrive/Desktop/best_models/best.pt"

    model = YOLO(custom_model_path)
    
    st.header('Computer Vision in Retail Web App')
    st.subheader('Models are pretrained on YOLO models')
    st.write('Welcome!')
    
    with st.form("my_form"):
        uploaded_file = st.file_uploader("Upload image or video", type=['mp4', 'jpg', 'jpeg', 'png'])
        st.form_submit_button(label='Submit')

    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            img = Image.open(uploaded_file)
            img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            results = model(img_cv2)
            img_with_boxes = draw_boxes(img_cv2, results)
            st.image(img_with_boxes, channels="BGR", use_column_width=True)

        elif uploaded_file.type.startswith('video'):
            file_binary = uploaded_file.read()
            with open('video.mp4', "wb") as temp_file:
                temp_file.write(file_binary)
            video_stream = cv2.VideoCapture('video.mp4')
            width = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            fps = int(video_stream.get(cv2.CAP_PROP_FPS))
            output_path = 'output.mp4'
            out_video = cv2.VideoWriter(output_path, int(fourcc), fps, (width, height))

            with st.spinner('Processing video...'):
                while True:
                    ret, frame = video_stream.read()
                    if not ret:
                        break
                    results = model(frame)
                    frame_with_boxes = draw_boxes(frame, results)
                    out_video.write(frame_with_boxes)
                video_stream.release()
                out_video.release()
            st.video(output_path)

if __name__ == "__main__":
    app()
