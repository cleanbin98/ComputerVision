import re
import cv2
import numpy as np
import tensorflow as tf

#머신 러닝 학습을 위한 학습 라벨 파일을 가져옴.
with open("./mscoco_complete_label_map.pbtxt", "rt") as f:
    pb_classes = f.read().rstrip("\n").split("\n")
    classes_label = dict()

    #학습 라벨 파일에 접근해서 필요한 데이터를 가져옴.
    for i in range(0, len(pb_classes), 5):
        pb_classId = int(re.findall("\d+", pb_classes[i + 2])[0])
        pattern = 'display_name: "(.*?)"'
        pb_text = re.search(pattern, pb_classes[i + 3])
        classes_label[pb_classId] = pb_text.group(1)

model = tf.saved_model.load("./ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model")
capture = cv2.VideoCapture("C:/ImageSamle/bird.mp4")

while True:
    ret, frame = capture.read()

    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        break
    #화면 출력을 위한 변수 선언.
    input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(frame, dsize=(1080,720), interpolation=cv2.INTER_LINEAR)
    input_tensor = tf.convert_to_tensor(input_img)
    input_tensor = input_tensor[tf.newaxis, ...]

    output_dict = model.signatures["serving_default"](input_tensor)

    classes = output_dict["detection_classes"][0]
    scores = output_dict["detection_scores"][0]
    boxes = output_dict["detection_boxes"][0]

    height, width, _ = input_img.shape
    #70% 이상 정확해야지 라벨의 데이터를 화면에 보여줌.
    for idx, score in enumerate(scores):
        if score > 0.7:
            class_id = int(classes[idx])
            box = boxes[idx]

            x1 = int(box[1] * width)
            y1 = int(box[0] * height)
            x2 = int(box[3] * width)
            y2 = int(box[2] * height)

            cv2.rectangle(input_img, (x1, y1), (x2, y2), 255, 1)
            cv2.putText(input_img, classes_label[class_id] + ":" + str(float(score)), (x1, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 1)

    cv2.imshow("Object Detection", input_img)
    if cv2.waitKey(33) == ord("q"):
        break
