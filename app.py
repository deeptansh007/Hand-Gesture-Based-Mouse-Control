import cv2
import numpy as np
import pyautogui
import time
from cvzone.HandTrackingModule import HandDetector
from collections import deque

class GestureController:
    def __init__(self):
        # Initialize camera with multiple backend attempts
        self.cap = self.initialize_camera()
        self.detector = HandDetector(detectionCon=0.85, maxHands=1)
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Mouse control
        self.smoothing = 7
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.prev_points = deque(maxlen=self.smoothing)
        
        # Click control
        self.click_thresh = 35
        self.click_cooldown = 0
        self.drag_thresh = 40
        self.is_dragging = False
        
        # Visual settings
        self.colors = {
            'index': (255, 100, 0),  # Orange
            'thumb': (0, 255, 100),  # Teal
            'click': (0, 0, 255),     # Red
            'drag': (255, 255, 0),    # Yellow
            'scroll': (255, 0, 255)   # Purple
        }

    def initialize_camera(self):
        """Try multiple camera indices and backends"""
        backends = [
            cv2.CAP_DSHOW,  # DirectShow (Windows)
            cv2.CAP_MSMF,   # Media Foundation (Windows)
            cv2.CAP_V4L2    # Video4Linux (Linux)
        ]
        
        for backend in backends:
            for i in range(3):  # Try indices 0-2
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    print(f"Camera found at index {i} with backend {backend}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    return cap
        raise IOError("Could not initialize camera")

    def map_to_screen(self, x, y, w, h):
        """Convert camera coordinates to screen coordinates with boundary checks"""
        screen_x = np.interp(x, [0, w], [0, self.screen_w])
        screen_y = np.interp(y, [0, h], [0, self.screen_h])
        return np.clip(screen_x, 0, self.screen_w), np.clip(screen_y, 0, self.screen_h)

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame read error - reinitializing camera...")
                    self.cap.release()
                    self.cap = self.initialize_camera()
                    time.sleep(1)
                    continue

                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                hands, _ = self.detector.findHands(frame, flipType=False)

                if hands:
                    hand = hands[0]
                    lmList = hand["lmList"]
                    fingers = self.detector.fingersUp(hand)

                    # Get finger positions
                    index_tip = lmList[8][:2]
                    thumb_tip = lmList[4][:2]
                    middle_tip = lmList[12][:2]

                    # Draw landmarks
                    cv2.circle(frame, index_tip, 15, self.colors['index'], cv2.FILLED)
                    cv2.circle(frame, thumb_tip, 15, self.colors['thumb'], cv2.FILLED)
                    cv2.circle(frame, middle_tip, 10, (0, 255, 255), cv2.FILLED)

                    # Smooth mouse movement
                    self.prev_points.append(index_tip)
                    if len(self.prev_points) == self.smoothing:
                        weights = np.linspace(0.5, 1.5, self.smoothing)
                        avg_x = int(np.average([p[0] for p in self.prev_points], weights=weights))
                        avg_y = int(np.average([p[1] for p in self.prev_points], weights=weights))
                        
                        screen_x, screen_y = self.map_to_screen(avg_x, avg_y, w, h)
                        
                        # Smoothed movement with acceleration
                        self.clocX = self.plocX + (screen_x - self.plocX) / 3
                        self.clocY = self.plocY + (screen_y - self.plocY) / 3
                        
                        pyautogui.moveTo(self.clocX, self.clocY, _pause=False)
                        self.plocX, self.plocY = self.clocX, self.clocY

                    # Gesture detection
                    thumb_index_dist = np.linalg.norm(np.array(index_tip) - np.array(thumb_tip))
                    index_middle_dist = np.linalg.norm(np.array(index_tip) - np.array(middle_tip))

                    # Right-click (middle finger down)
                    if fingers[2] == 0 and self.click_cooldown == 0:
                        pyautogui.rightClick()
                        cv2.line(frame, index_tip, middle_tip, self.colors['scroll'], 3)
                        self.click_cooldown = 20
                    
                    # Left-click (pinch)
                    elif thumb_index_dist < self.click_thresh:
                        if self.click_cooldown == 0:
                            if not self.is_dragging:
                                pyautogui.click()
                                cv2.line(frame, index_tip, thumb_tip, self.colors['click'], 3)
                            else:
                                cv2.line(frame, index_tip, thumb_tip, self.colors['drag'], 3)
                            self.click_cooldown = 15
                    
                    # Drag (sustained pinch)
                    elif thumb_index_dist < self.drag_thresh:
                        if not self.is_dragging:
                            pyautogui.mouseDown()
                            self.is_dragging = True
                            cv2.line(frame, index_tip, thumb_tip, self.colors['drag'], 3)
                    else:
                        if self.is_dragging:
                            pyautogui.mouseUp()
                            self.is_dragging = False
                        if self.click_cooldown > 0:
                            self.click_cooldown -= 1

                # Display UI
                cv2.putText(frame, "Move: Index Finger", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Left-Click: Pinch Index-Thumb", (10, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Right-Click: Lower Middle Finger", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Drag: Hold Pinch While Moving", (10, 135), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press 'Q' to Quit", (10, h-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Advanced Hand Gesture Control", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            if self.is_dragging:
                pyautogui.mouseUp()

if __name__ == "__main__":
    controller = GestureController()
    controller.run()