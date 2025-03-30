import pygame
import cv2
import mediapipe as mp

# Initialize Pygame
pygame.init()

# Define screen dimensions (16:9 aspect ratio)
WIDTH, HEIGHT = 960, 540
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Eyebrow Raise Ripple Effect")

# Initialize MediaPipe Hand Tracking & Face Mesh
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Open webcam (16:9 aspect ratio)
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)   # Set width
cap.set(4, HEIGHT)  # Set height

# Circle properties
circle_x, circle_y = WIDTH // 2, HEIGHT // 2
circle_radius = 10
circle_color = (0, 255, 0)  # Green

# Ripple effect list
ripples = []

def detect_eyebrow_raise(landmarks):
    """Detects if both eyebrows are raised"""
    left_eyebrow = landmarks[70].y  # Adjusted landmark for better accuracy
    left_eye_top = landmarks[159].y
    right_eyebrow = landmarks[300].y
    right_eye_top = landmarks[386].y

    # Compare eyebrow-to-eye height difference
    left_ratio = left_eye_top - left_eyebrow
    right_ratio = right_eye_top - right_eyebrow

    return left_ratio > 0.03 and right_ratio > 0.03  # Adjusted threshold

running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen

    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results_hands = hands.process(rgb_frame)
    results_face = face_mesh.process(rgb_frame)

    # Detect hands and move circle
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger tip coordinates (landmark 8)
            index_tip = hand_landmarks.landmark[8]

            # Convert normalized coordinates (0 to 1) to screen coordinates
            circle_x = int(index_tip.x * WIDTH)
            circle_y = int(index_tip.y * HEIGHT)

    # Detect eyebrow raise and create ripple effect
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            if detect_eyebrow_raise(face_landmarks.landmark):
                ripples.append([circle_x, circle_y, 20])  # Add a new ripple

    # Draw ripple effects
    new_ripples = []
    for ripple in ripples:
        x, y, r = ripple
        pygame.draw.circle(screen, (0, 0, 255), (x, y), r, 2)
        if r < 100:  # Expand ripple
            new_ripples.append([x, y, r + 5])

    ripples = new_ripples  # Update ripples

    # Draw the main circle
    pygame.draw.circle(screen, circle_color, (circle_x, circle_y), circle_radius)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.update()

    # Show OpenCV window (for debugging)
    cv2.imshow("Hand & Eyebrow Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()
