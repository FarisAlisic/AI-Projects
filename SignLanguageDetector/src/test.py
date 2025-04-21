#   cd /Users/farisalicic/Desktop/programming/embeded
#   source asl_env/bin/activate 

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QVBoxLayout, QHBoxLayout, QWidget, QSizePolicy)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from mediapipe import solutions
import tensorflow as tf

class ASLApp(QMainWindow):
    def __init__(self, model_path):
        super().__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                          'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
        
        # MediaPipe setup
        self.mp_hands = solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = solutions.drawing_utils
        
        # UI Setup
        self.setWindowTitle("ASL Letter Detector")
        self.setFixedSize(1200, 600)  # Fixed window size
        
        # Main widget
        self.main_widget = QWidget()
        self.layout = QHBoxLayout()
        self.main_widget.setLayout(self.layout)
        self.setCentralWidget(self.main_widget)
        
        # Left panel (camera and prediction)
        left_panel = QWidget()
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Camera feed with fixed size
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFixedSize(800, 450)  # Fixed camera display size
        self.camera_label.setStyleSheet("border: 2px solid black;")
        left_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)
        
        # Prediction display
        self.prediction_label = QLabel("Show your hand to begin")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFixedHeight(80)  # Fixed height
        self.prediction_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.prediction_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #2c3e50;"
            "background-color: #ecf0f1; border: 2px solid #3498db;"
            "padding: 10px; border-radius: 10px;"
        )
        left_layout.addWidget(self.prediction_label)
        
        # Right panel (reference)
        right_panel = QWidget()
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        # ASL reference image with fixed size
        self.reference_label = QLabel()
        self.reference_label.setAlignment(Qt.AlignCenter)
        self.reference_label.setFixedSize(350, 550)  # Fixed reference image size
        self.reference_label.setPixmap(QPixmap("asl.jpeg").scaled(
            350, 550, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.reference_label.setStyleSheet("border: 2px solid black;")
        right_layout.addWidget(self.reference_label, alignment=Qt.AlignCenter)
        
        # Add panels to main layout
        self.layout.addWidget(left_panel)
        self.layout.addWidget(right_panel)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 30 FPS
        
    def process_frame(self, frame):
        """Process frame and return landmarks"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # Extract landmarks
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for hand in results.multi_hand_landmarks
                for lm in hand.landmark
            ]).flatten()
            
            return frame, landmarks
        return frame, None
    
    def update_frame(self):
        """Update camera feed and make predictions"""
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            processed_frame, landmarks = self.process_frame(frame)
            
            # Convert to QImage
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Scale to fit the fixed label size while maintaining aspect ratio
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.camera_label.width(), self.camera_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            
            # Continuous prediction
            if landmarks is not None:
                pred = self.model.predict(np.array([landmarks]), verbose=0)
                letter = self.class_names[np.argmax(pred)]
                confidence = np.max(pred)
                self.prediction_label.setText(
                    f"Predicted: {letter}\nConfidence: {confidence:.1%}"
                )
    
    def closeEvent(self, event):
        """Clean up on window close"""
        self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Load ASL reference image (replace with your path)
    # You'll need an "asl.jpeg" file in the same directory
    window = ASLApp("models/asl_model.h5")
    window.show()
    sys.exit(app.exec_())