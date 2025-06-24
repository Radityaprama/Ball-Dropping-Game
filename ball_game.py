import cv2
import mediapipe as mp
import pygame
import sys
import numpy as np

# Inisialisasi Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi Pygame
pygame.init()
WIDTH, HEIGHT = 640, 480
win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ball Dropping Game - Dual Hand Detection")
clock = pygame.time.Clock()

# Ball settings
ball1_x = WIDTH // 3
ball2_x = WIDTH * 2 // 3
ball1_y = 0
ball2_y = 0
ball_radius = 20
ball_speed = 5

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)

font = pygame.font.SysFont('Arial', 30)
score = 0

running = True
while running:
    clock.tick(30)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    ret, frame = cap.read()
    if not ret:
        continue

    # frame = cv2.flip(frame, 1) # Optional: uncomment kalau mau mirror
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    index_fingers = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            index_tip = handLms.landmark[8]  # ujung telunjuk
            x = int(index_tip.x * WIDTH)
            y = int(index_tip.y * HEIGHT)
            index_fingers.append((x, y))

            # Ini baru bener: gambar landmark langsung di frame (BGR)
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Bola jatuh
    ball1_y += ball_speed
    ball2_y += ball_speed

    if ball1_y + ball_radius > HEIGHT:
        ball1_y = 0
    if ball2_y + ball_radius > HEIGHT:
        ball2_y = 0

    # Deteksi tabrak bola
    if len(index_fingers) >= 1:
        fx1, fy1 = index_fingers[0]
        dist1 = ((fx1 - ball1_x) ** 2 + (fy1 - ball1_y) ** 2) ** 0.5
        if dist1 <= ball_radius:
            ball1_y = 0
            score += 1

    if len(index_fingers) >= 2:
        fx2, fy2 = index_fingers[1]
        dist2 = ((fx2 - ball2_x) ** 2 + (fy2 - ball2_y) ** 2) ** 0.5
        if dist2 <= ball_radius:
            ball2_y = 0
            score += 1

    # convert frame ke pygame surface
    frame = np.rot90(frame)  # langsung dari frame (BGR, warna natural)
    frame_surface = pygame.surfarray.make_surface(frame)

    # Gambar background kamera
    win.blit(frame_surface, (0, 0))

    # Gambar bola
    pygame.draw.circle(win, (255, 0, 0), (ball1_x, ball1_y), ball_radius)
    pygame.draw.circle(win, (0, 0, 255), (ball2_x, ball2_y), ball_radius)

    # Gambar score
    score_text = font.render(f'Score: {score}', True, (255, 255, 0))
    win.blit(score_text, (10, 10))

    pygame.display.update()

cap.release()
pygame.quit()
sys.exit()

