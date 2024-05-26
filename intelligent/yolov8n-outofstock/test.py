from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('train6/weights/best.pt')
    metrics = model.val(data="config.yaml", save=True)