from ultralytics import YOLO
import cv2
import os

output_dir = r'C:\Users\USER\Documents\Projects\object_detector\output_videos'

def process_video_yolov8(input_path, output_path):
    try:
        model = YOLO('yolov8n.pt')  # Or a larger model like 'yolov8m.pt' for better accuracy
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)  # Predict objects in the frame
            annotated_frame = results[0].plot() # Visualize the results

            out.write(annotated_frame)

        cap.release()
        out.release()
        print(f"Processed video saved to: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    video_path = input("Enter the full path to your video file: ")
    base_name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"processed_{name_without_ext}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True) # Ensure the output directory exists
    process_video_yolov8(video_path, output_path)
    print(f"Processed video saved as: {output_path}")