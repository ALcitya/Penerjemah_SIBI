import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=0,
    min_detection_confidence=0.5
)

mp_hands = mp.solutions.hands
hands_fallback = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

POSE_INDEKS_RELEVAN = [11, 12, 13, 14, 15, 16, 23, 24]

def crop_hand_from_frame(frame, target_size=(160, 160), padding=40):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)
    h, w, _ = frame.shape

    x_list, y_list = [], []

    if results.pose_landmarks:
        for idx in POSE_INDEKS_RELEVAN:
            lm = results.pose_landmarks.landmark[idx]
            if lm.visibility > 0.3:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

    # fallback: Holistic gagal total -> coba deteksi tangan langsung
    if len(x_list) == 0:
        fallback_results = hands_fallback.process(frame_rgb)
        if fallback_results.multi_hand_landmarks:
            for hand_landmarks in fallback_results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_list.append(int(lm.x * w))
                    y_list.append(int(lm.y * h))

    if len(x_list) == 0:
        return None

    x_min = max(min(x_list) - padding, 0)
    y_min = max(min(y_list) - padding, 0)
    x_max = min(max(x_list) + padding, w)
    y_max = min(max(y_list) + padding, h)

    cropped = frame[y_min:y_max, x_min:x_max]
    if cropped.shape[0] > 0 and cropped.shape[1] > 0:
        return cv2.resize(cropped, target_size)
    return None