import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.1,
    min_tracking_confidence=0.1
)
# crop hand from
def crop_hand_from_frame(frame, target_size=(160,160), padding=40):
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)
    
    if not result.multi_hand_landmarks:
        return None
    h,w, _ = frame.shape
    x_list, y_list = [], []
    # gabung semua tangan
    for hand_landmarks in result.multi_hand_landmarks:
        for lm in hand_landmarks.landmark:
            x_list.append(int(lm.x * w))
            y_list.append(int(lm.y * h))
            
    x_min = max(min(x_list) - padding, 0)
    y_min = max(min(y_list) - padding, 0)
    x_max = min(max(x_list) + padding, w)
    y_max = min(max(y_list) + padding, h)

    cropped_hand = frame[y_min:y_max, x_min:x_max]
    if cropped_hand.shape[0] > 0 and cropped_hand.shape[1]>0:
        return cv2.resize(cropped_hand, target_size)
    return None