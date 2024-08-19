import cv2
import mediapipe as mp
import pyautogui
import math

# Inisialisasi MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Dapatkan ukuran layar
screen_width, screen_height = pyautogui.size()

cap = cv2.VideoCapture(0)

# Threshold untuk mendeteksi klik dan drag
CLICK_THRESHOLD = 30  # Jarak dalam piksel untuk mendeteksi klik

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

dragging = False  # Flag untuk mendeteksi apakah sedang melakukan drag

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Membalik gambar horizontal untuk efek mirror
    img = cv2.flip(img, 1)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for lm in hand_landmarks.landmark]
            
            # Ambil posisi jari telunjuk dan ibu jari
            index_finger_tip = landmarks[8]  # Ujung jari telunjuk
            thumb_tip = landmarks[4]         # Ujung jari ibu jari
            
            # Konversi posisi jari ke koordinat layar
            screen_index_finger_tip = (
                int((index_finger_tip[0] / img.shape[1]) * screen_width),
                int((index_finger_tip[1] / img.shape[0]) * screen_height)
            )
            screen_thumb_tip = (
                int((thumb_tip[0] / img.shape[1]) * screen_width),
                int((thumb_tip[1] / img.shape[0]) * screen_height)
            )
            
            # Hitung jarak antara jari telunjuk dan ibu jari
            distance = calculate_distance(index_finger_tip, thumb_tip)
            
            if distance < CLICK_THRESHOLD:
                if not dragging:
                    # Mulai drag (klik kiri tahan)
                    pyautogui.mouseDown()
                    dragging = True
                # Gerakkan kursor selama drag
                pyautogui.moveTo(screen_index_finger_tip)
            else:
                if dragging:
                    # Lepaskan drag (drop)
                    pyautogui.mouseUp()
                    dragging = False
            
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
