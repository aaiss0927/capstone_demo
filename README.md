# 사용법
```
git clone https://github.com/aaiss0927/capstone_demo.git
cd capstone_demo/YOLO
pip install -r requirements.txt
```

- 모델(ema_cleaned_class_2.pt) 다운로드 후 capstone_demo/YOLO/weights/ema_cleaned_class_2.pt 경로로 위치시킬 것 </br>
(링크: https://drive.google.com/file/d/1A8auSAImIEBczMUwk4jahiPZDf49EwL-/view?usp=drive_link)

- capstone_demo/YOLO//demo/images/inference/___.jpg 경로에 추론할 이미지 업로드

```
python yolo/inference.py --image_path /demo/images/inference/0087_FL_FWW_00003.jpg
```

추론 이미지 경로: /shared/home/kdd/HZ/capstone/YOLO/output.jpg
