import cv2

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

# print("object list")
# print(classes)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)

# def click_button(event, x, y, flags,params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         print(x, y)

cv2.namedWindow("Frame")

while True:
    ret, frame = cap.read()

    (class_ids, scores, bboxes) = model.detect(frame)
    for class_ids, scores, bboxes in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bboxes
        class_name = classes[class_ids]

        cv2.putText(frame, str(class_name), (x, y - 20), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

    # print("class ids", class_ids)
    # print("scores", scores)
    # print("bboxes", bboxes)

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
