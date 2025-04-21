import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from mediapipe import solutions
from sklearn.metrics import classification_report

class ASLProcessor:
    def __init__(self):
        self.mp_hands = solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.7
        )
        self.class_map = {
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 
            'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
            'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24,
            'Z': 25, 'space': 26, 'del': 27, 'nothing': 28
        }
        self.reverse_map = {v: k for k, v in self.class_map.items()}

    def load_train_data(self, data_dir):
        X, y = [], []
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                for file in os.listdir(class_dir):
                    if file.endswith('.npy'):
                        data = np.load(os.path.join(class_dir, file))
                        if data.shape == (63,):
                            X.append(data)
                            y.append(self.class_map[class_name])
        return np.array(X), np.array(y)

    def process_test_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        results = self.mp_hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            return np.array([
                [lm.x, lm.y, lm.z] 
                for hand in results.multi_hand_landmarks
                for lm in hand.landmark
            ]).flatten()
        return None

    def evaluate_model(self, model, test_dir):
        X_test, y_true = [], []
        for file in os.listdir(test_dir):
            if file.lower().endswith(('.jpg', '.jpeg')):
                # Extract true label from filename (format: "letter_test.jpg")
                true_label = file.split('_')[0].lower()
                if true_label == 'del':
                    true_label = 'del'
                elif true_label == 'space':
                    true_label = 'space'
                elif true_label == 'nothing':
                    true_label = 'nothing'
                else:
                    true_label = true_label.upper()  # For letters
                
                landmarks = self.process_test_image(os.path.join(test_dir, file))
                if landmarks is not None and landmarks.shape == (63,):
                    X_test.append(landmarks)
                    y_true.append(self.class_map.get(true_label, -1))  # -1 for unknown
        
        if not X_test:
            raise ValueError("No valid test images found!")
        
        X_test = np.array(X_test)
        y_true = np.array(y_true)
        
        # Filter out unknown labels
        valid_idx = y_true != -1
        X_test = X_test[valid_idx]
        y_true = y_true[valid_idx]
        
        # Predictions
        y_pred = np.argmax(model.predict(X_test), axis=1)
        
        # Generate report
        target_names = [self.reverse_map[i] for i in sorted(np.unique(y_true))]
        print("\n=== Test Results ===")
        print(classification_report(
            y_true, y_pred, 
            target_names=target_names,
            zero_division=0
        ))
        
        return y_true, y_pred

# Initialize
processor = ASLProcessor()

# 1. Load Training Data
X_train, y_train = processor.load_train_data('data/train')
y_train = utils.to_categorical(y_train, len(processor.class_map))

# 2. Build and Train Model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(63,)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(processor.class_map), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=30, batch_size=32)

# 3. Evaluate on Test JPEGs
y_true, y_pred = processor.evaluate_model(model, 'data/test')

# 4. Save Model
os.makedirs('models', exist_ok=True)
model.save('models/asl_model.h5')
print("Model saved to models/asl_model.h5")