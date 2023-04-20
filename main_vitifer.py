from ultralytics import YOLO
import cv2



def grape_detection_davide():
    # Load a model
    model = YOLO('yolov8n.pt')  # load an official detection model
    model = YOLO('/home/mmt-ben/YOLOV8_VITIFER/models/grape_detection/best.pt')  # load a custom model



    # Track with the model
    results = model.track(source="/home/mmt-ben/YOLOV8_VITIFER/media/my_video (1).mp4", show=True)



# Load a model
def train():
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    model.train(data='/home/mmt-ben/YOLOV8_VITIFER/ultralytics/datasets/bud.yaml', epochs=100, imgsz=640)
    metrics = model.val()  # evaluate model performance on the validation set
    #test on a video

def test():
    #model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
    new_mode_trained = "/home/mmt-ben/YOLOV8_VITIFER/runs/detect/train4/weights/best.pt"
    model = YOLO(new_mode_trained)
    results = model.track(source="/home/mmt-ben/YOLOV8_VITIFER/media/bud1.mkv", show=True)



test()