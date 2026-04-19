import cv2

for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print(i, ok)
    if ok:
        ret, frame = cap.read()
        print("  read:", ret, None if frame is None else frame.shape)
    cap.release()
    