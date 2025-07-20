import cv2

from detect_face import get_face_from_frame
from skin_analysis import analyze_skin
from recommend import recommend_products

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame, faces = get_face_from_frame(frame)

    for face in faces:
        concern = analyze_skin(face)
        products = recommend_products(concern)

        print(f"Detected Concern: {concern}")
        print(f"Recommended Products: {products}")

    cv2.imshow("Skin Concern Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
