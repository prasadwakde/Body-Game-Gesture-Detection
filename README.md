## Gesture Control Gaming System

This project implements a **real-time body gesture–controlled gaming system** using computer vision and machine learning. A webcam is used to capture a player’s body movements, extract pose landmarks, classify gestures, and map them to keyboard inputs that control games. The system removes the need for traditional input devices such as keyboards or controllers and enables players to interact with games using natural body movements.

The project explores **two gesture-based gaming setups**:

- A **4-gesture control system** for _Temple Run_
- A **10-gesture control system** for _Karate Fighter_

Pose estimation is performed using **MediaPipe Pose** and it detects the movements in 4 gesture game. For 10 gesture game, multiple machine learning models were evaluated. Based on real-time performance and accuracy, **K-Nearest Neighbors (KNN)** was selected as the final model, achieving approximately **90% accuracy** in live gameplay.

---

## 4-Gesture Game – Temple Run

The 4-gesture setup focuses on simple, intuitive body movements to control the _Temple Run_ game. This version emphasizes low latency and ease of use.

**Try the game yourself**  
[Temple Run on Poki](https://poki.com/en/g/temple-run-2)

**Gestures Used**

- Jump
- Duck
- Move Left
- Move Right

Each gesture is detected using pose landmarks and mapped directly to keyboard inputs required by the game. This setup demonstrates real-time gesture control with minimal gesture complexity.

---

## 10-Gesture Game – Karate Fighter

The 10-gesture setup controls the _Karate Fighter_ game and supports more complex movements, including punches, kicks, and combo actions.

**Try the game yourself**  
[Karate Fighter on Poki](https://poki.com/en/g/karate-fighter?msockid=3cd585d569666c481db3901368666d1a)

**Game Concept**  
The system captures full-body movements through a webcam, classifies gestures such as punches, kicks, crouches, and combos, and simulates corresponding keyboard presses in real time.

**Controls** 

The system maps the following physical gestures to keyboard keys:

| Gesture | Key Mapped | Action |
| ---------- | ---------- | ---------- |
| Extend both arms to the sides|W|Character Jumps|
| Crouch|S|Character Crouches|
| Extend Arm to Left|A|Move Left|
| Extend Arm to Right|D|Move Right|
| Make a Fist with Left or Right|L|Low Punch|
| Make a Fist with both Left and Right|I|High Punch|
| Lift and Extend Right Leg Forward|K|Strong Kick Attack|
| Lift Left Leg just above ground| J |High Kick Attack|
| Wakanda Forever| U |Combo Hit|
| Stand straight with Arms Extended down|-|No Move, Stays Idle|
---

## Project Folder Overview

- **4_gestures_finalCode**  
  Contains the final implementation for the Temple Run game using 4 basic gestures.

- **10_gestures_FinalCode**  
  Contains the final implementation for the Karate Fighter game using 10 gestures.

- **Models**  
  Contains experiments with multiple machine learning models:
  - Decision Trees
  - Support Vector Machine (SVM)
  - XGBoost / Extra Trees
  - Random Forest  
    The final system uses **KNN**, as it achieved the best real-time performance with approximately **90% accuracy**.

- **Results**  
  Contains live demo recordings, gameplay videos, and outputs of the working gesture-controlled system.

- **Python Scripts**
  - `optimized_collection.py` – Gesture data collection
  - `optimized_trainer.py` – Model training
  - `gameplay_with_KNN.py` – Real-time gameplay controller

---

## Prerequisites

Ensure Python is installed along with the required libraries:

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn joblib pydirectinput
```

---

# Usage Instructions

## Step 1: Data Collection

```bash
python optimized_collection.py
```

## Step 2: Train the Model

```bash
python optimized_trainer.py
```

## Step 3: Run the Controller

```bash
python gameplay_with_KNN.py
```

- Click on the game window once
- Control the character using body gestures

#  Demonstration Video
▶️ [Watch Demo Video](https://drive.google.com/file/d/1eAFeLSRwX_myB8Nm9I6voXBK2ro8zClB/view?usp=sharing)



# Troubleshooting

- Camera not opening? Ensure no other application is using the webcam

- Model not found? Verify optimized_karate_model.pkl exists

- Low accuracy? Record more diverse gesture samples and retrain the model

# Participants

- Glenn Paul Aby
- Prasad Deepak Wakde
- Samrudhi Ramesh Rao
