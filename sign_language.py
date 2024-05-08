import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip = 4

finger_fold_status = []

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # Accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append((lm.x, lm.y))

            # Draw circles around fingertips
            for tip in finger_tips:
                tip_x = int(lm_list[tip][0] * w)
                tip_y = int(lm_list[tip][1] * h)
                cv2.circle(img, (tip_x, tip_y), 10, (255, 0, 0), -1)

            # Check if fingers are folded
            fold_status = [lm_list[tip][0] < lm_list[tip - 1][0] for tip in finger_tips]
            finger_fold_status.append(all(fold_status))

            # Check if all fingers are folded
            if all(finger_fold_status):
                # Check if thumb is raised up or down
                thumb_tip_y = lm_list[thumb_tip][1]
                previous_thumb_tip_y = lm_list[thumb_tip][1]  # You need to update this value in a real-time scenario
                if thumb_tip_y < previous_thumb_tip_y:
                    print("LIKE")
                    cv2.putText(img, "LIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    print("DISLIKE")
                    cv2.putText(img, "DISLIKE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
                                   mp_draw.DrawingSpec((0,255,0),4,2))

    cv2.imshow("hand tracking", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()