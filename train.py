from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

train_results = model.train(
    data = 'D:/Projects/ComputerVisionProjects/forestFireDetect/datasets/data.yaml',
    epochs = 200,  
    imgsz = 640,
    device = '0',  # Use GPU 0
    workers = 0
)

metrics = model.val() 
