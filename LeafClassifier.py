import os
import ultralytics
from ultralytics import YOLO

pt = os.path.join(os.path.dirname(__file__), "model.pt")
if not os.path.exists(pt):
  pt = input("Введите путь к модели: ")

files = os.path.join(os.path.dirname(os.path.abspath(__file__)), "files")
if not os.path.exists(files):
  files = input("Введите путь к файлу или папке с изображениями или видео: ")

script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "results")
os.makedirs(save_dir, exist_ok=True)

ultralytics.checks()

model = YOLO(pt)

results = model.predict(source=files, save=True, save_txt=True, save_conf=True, project = save_dir, name="prediction")

for result in results:
    print(result.boxes)