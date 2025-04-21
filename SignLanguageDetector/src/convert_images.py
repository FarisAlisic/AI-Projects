import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# Configuration
INPUT_BASE = "raw_images"
OUTPUT_BASE = "data"
DEBUG = True  # Set to False to disable verbose logging

def log(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

def setup_folders():
    """Ensure required folder structure exists"""
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    if not os.path.exists(INPUT_BASE):
        raise FileNotFoundError(f"Input directory '{INPUT_BASE}' not found")
    log(f"Base directories verified")

def get_all_classes():
    """Find all unique classes in train/test folders"""
    classes = set()
    
    for split in ['train', 'test']:
        split_path = os.path.join(INPUT_BASE, split)
        if os.path.exists(split_path):
            log(f"Scanning {split_path}")
            classes.update(
                d for d in os.listdir(split_path) 
                if not d.startswith('.') 
                and os.path.isdir(os.path.join(split_path, d))
            )
    
    if not classes:
        raise ValueError(f"No class folders found in {INPUT_BASE}")
    
    log(f"Found classes: {sorted(classes)}")
    return sorted(classes)

def process_image(image_path):
    try:
        log(f"Processing {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            log(f"Could not read {image_path}")
            return None
        
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            landmarks = np.array([
                [lm.x, lm.y, lm.z] 
                for hand in results.multi_hand_landmarks
                for lm in hand.landmark
            ]).flatten()
            log(f"Successfully processed {image_path}")
            return landmarks
        log(f"No hands detected in {image_path}")
        return None
    except Exception as e:
        log(f"Error processing {image_path}: {str(e)}")
        return None

def convert_class(cls):
    """Process all images for a single class"""
    processed = 0
    skipped = 0
    failed = 0
    
    for split in ['train', 'test']:
        input_dir = os.path.join(INPUT_BASE, split, cls)
        output_dir = os.path.join(OUTPUT_BASE, split, cls)
        
        if not os.path.exists(input_dir):
            log(f"Skipping missing {input_dir}")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        image_files = [
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        log(f"Found {len(image_files)} images in {input_dir}")
        
        for img_file in image_files:
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, f"{os.path.splitext(img_file)[0]}.npy")
            
            if os.path.exists(output_path):
                skipped += 1
                continue
                
            landmarks = process_image(input_path)
            if landmarks is not None:
                np.save(output_path, landmarks)
                processed += 1
            else:
                failed += 1
    
    return processed, skipped, failed

if __name__ == "__main__":
    print("=== ASL Image Conversion Debug Mode ===")
    setup_folders()
    
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.6
    )
    
    classes = get_all_classes()
    total_stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    
    for cls in classes:
        print(f"\nProcessing class: {cls}")
        stats = convert_class(cls)
        total_stats['processed'] += stats[0]
        total_stats['skipped'] += stats[1]
        total_stats['failed'] += stats[2]
        print(f"  Processed: {stats[0]}, Skipped: {stats[1]}, Failed: {stats[2]}")
    
    print("\n=== Conversion Summary ===")
    print(f"Total Processed: {total_stats['processed']}")
    print(f"Total Skipped (already existed): {total_stats['skipped']}")
    print(f"Total Failed: {total_stats['failed']}")
    print(f"\nOutput saved to: {os.path.abspath(OUTPUT_BASE)}")