from ultralytics import YOLO
import torch


def main():
    model = YOLO("yolov8n.pt")
    model.train(
        data=r"C:\Users\eren1\PycharmProjects\pythonProject1\venv\ua_detrra_cdataset\data.yaml",
        imgsz=416,
        epochs=50,
        batch=16,
        augment=True
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()