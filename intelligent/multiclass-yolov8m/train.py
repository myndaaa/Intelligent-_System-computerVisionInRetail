from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8m.pt')  # load a custom model

    # parameters
    
    model.train(data="config.yaml",
                save=True,
                epochs=200,
                batch=16,
                imgsz=640,
                model="yolov8m.yaml",
                lrf=0.01,
                lr0=0.001,
                save_period=5
                )
