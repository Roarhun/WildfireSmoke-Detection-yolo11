from ultralytics import YOLO
import cv2
model = YOLO('detectSmoke.pt')  

# results = model('datasets/test/images/ck0kosu846k350794gya7bot0_jpeg.rf.e980ecef63c864103fc6d8f4d8ed7ded.jpg')  # Predict on an image
# results[0].show()  

video_path = 'video_test.mp4'  # หรือ 0 ถ้าใช้ webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # ตรวจจับวัตถุในเฟรม
    results = model(frame)

    # แสดงผลลัพธ์ที่มี bounding boxes
    annotated_frame = results[0].plot()

    # แสดงเฟรม
    cv2.imshow("Smoke Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()