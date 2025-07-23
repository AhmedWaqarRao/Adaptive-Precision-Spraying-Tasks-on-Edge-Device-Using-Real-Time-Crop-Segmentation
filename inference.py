import torch
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO

model_path = r"C:\Users\ahmed\Downloads\last (6).pt"
image_path = r"A:\3dim\tobacco_01\tobacco_0043.jpg"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(model_path).to(device)

results = model(image_path, imgsz=640)
res_plotted = results[0].plot()
cv.imshow("YOLO v8 segmentation", res_plotted)
cv.waitKey(0)
cv.destroyAllWindows()