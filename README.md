# 사용법
```
git clone https://github.com/aaiss0927/capstone_demo.git
cd YOLO
pip install -r requirements.txt
```

모델(ema_cleaned_class_2.pt) 다운로드 후 /shared/home/kdd/HZ/capstone/YOLO/weights/ema_cleaned_class_2.pt 경로로 위치시킬 것 </br>
(링크: https://drive.google.com/file/d/1A8auSAImIEBczMUwk4jahiPZDf49EwL-/view?usp=drive_link)

```
python yolo/inference_single.py --image_path /shared/home/kdd/HZ/capstone/YOLO/demo/images/inference/0087_FL_FWW_00003.jpg
```

추론 이미지 경로: /shared/home/kdd/HZ/capstone/YOLO/output.jpg
