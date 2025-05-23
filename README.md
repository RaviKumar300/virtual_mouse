# Virtual Cursor using Hand Tracking & Eyebrow Gestures

## Overview
This project implements a virtual cursor using hand tracking and eyebrow gestures, powered by OpenCV, MediaPipe, and Pygame. The user can control the cursor with their index finger, and raising both eyebrows triggers a ripple effect at the cursor's position.

## Features
- **Hand Tracking:** Uses MediaPipe Hands to track the index finger for cursor movement.
- **Eyebrow Detection:** Uses MediaPipe Face Mesh to detect eyebrow raises.
- **Ripple Effect:** Creates a visual ripple effect when the user raises both eyebrows.
- **Real-time Processing:** Works with a live webcam feed.

## Technologies Used
- Python
- OpenCV
- MediaPipe
- Pygame

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/RaviKumar300/virtual_cursor.git
   cd virtual_cursor
   ```
2. Install dependencies:
   ```sh
   pip install opencv-python mediapipe pygame numpy
   ```
3. Run the script:
   ```sh
   python virtual_cursor.py
   ```

## How It Works
1. **Cursor Movement:** The index finger tip's position is tracked and used to move a cursor on the screen.
2. **Eyebrow Raise Detection:** If both eyebrows are raised, a ripple effect is triggered at the cursor's location.
3. **Visual Feedback:** The webcam feed shows hand landmarks, and Pygame renders the cursor and ripples.

## Controls
- Move your **index finger** to control the cursor.
- Raise **both eyebrows** to create a ripple effect.
- Press `q` to exit the OpenCV window.
- Click the close button to exit the Pygame window.

## Future Improvements
- Add gesture-based clicking functionality.
- Improve accuracy of eyebrow detection.
- Implement a smoother cursor movement algorithm.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Developed by Ravi Kumar.

