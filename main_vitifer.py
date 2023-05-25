from ultralytics import YOLO
import cv2



new_mode_trained = "/home/mmt-ben/YOLOV8_VITIFER/runs/detect/train9/weights/best.pt"
video1_path = "/home/mmt-ben/YOLOV8_VITIFER/media/bud1.mkv"
video_path2 = "/home/mmt-ben/YOLOV8_VITIFER/media/VID_20230330_104555.mp4"




def resize_image(img, percentage):
    # Display the frame
    # saved in the file
    scale_percent = 70  # percent of original size
    width = int(img.shape[1] * percentage / 100)
    height = int(img.shape[0] * percentage / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized



def grape_detection_davide():
    # Load a model
    #model = YOLO('yolov8n.pt')  # load an official detection model
    model = YOLO('/home/mmt-ben/YOLOV8_VITIFER/models/grape_detection/best.pt')  # load a custom model
    # Track with the model
    results = model.track(source="/home/mmt-ben/YOLOV8_VITIFER/media/my_video (1).mp4", show=True)



# Load a model
def train():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data='/home/mmt-ben/YOLOV8_VITIFER/ultralytics/datasets/bud_e_l.yaml', epochs=100, imgsz=640)

    #validation
    metrics = model.val()  # evaluate model performance on the validation set
    #test on a video

def test():
    #model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    model = YOLO(new_mode_trained)
    results = model.track(source=video_path2, show=True)

def test_and_save_video():
    model = YOLO(new_mode_trained)

    # Open the video file

    cap = cv2.VideoCapture(video_path2)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)
    fps = float(cap.get(cv2.CAP_PROP_FPS))


    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out = cv2.VideoWriter("test.avi",fourcc, fps, size)  # setta la giusta risoluzionw e fps

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()


            # Display the annotated frame
            #annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame)
            #cv2.imwrite("img", annotated_frame)
            annotated_frame = resize_image(annotated_frame, 50)
            cv2.imshow("YOLOv8 Inference", annotated_frame)


            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    out.release()
    cap.release()
    cv2.destroyAllWindows()



#test_and_save_video()
test()
#train()

