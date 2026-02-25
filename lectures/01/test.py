from ultralytics import YOLO
model = YOLO('yolo26n.pt')

#Perform object detection on an imagels
results = model('i01.jpg', device = "cpu")
results[0].show()