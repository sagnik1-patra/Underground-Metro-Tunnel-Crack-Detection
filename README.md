🚇 MetroSense – Hybrid AIS + GWO Optimized Underground Metro Tunnel Crack Detection System.

Dataset: Tzika et al. (Tunnel Inspection PDF → Extracted Frames),
Optimizer: Hybrid AIS (Artificial Immune System) + GWO (Grey Wolf Optimizer),
Model Outputs Prefixed With: ais_gwo_

📌 Project Overview-

MetroSense is an AI-powered tunnel crack detection and analysis system built for underground metro tunnels, subway networks, and railway shafts.
It extracts frames from the Tzika et al. metro tunnel inspection PDF, preprocesses them, and uses a CNN model optimized with a Hybrid AIS + GWO algorithm to classify structural cracks.

The system performs:

✔ Crack vs Non-Crack Classification
✔ Automated Hyperparameter Optimization (AIS + GWO)
✔ Model Training, Validation, Testing
✔ Multiple Graphs (Accuracy, Loss, Prediction, Comparison, Confusion Matrix)
✔ Full Output Saving (CSV, JSON, PNG, H5, PKL, YAML)
✔ Real-time useful insights for tunnel safety engineering
🎯 Why Hybrid AIS + GWO?
AIS (Artificial Immune System)

Clones best HPs,

Mutates them to explore better solutions

Good for local search

GWO (Grey Wolf Optimizer)

Packs wolves into Alpha, Beta, Delta

Updates positions towards best solutions

Excellent for global search and convergence

Hybrid AIS + GWO Advantages

✔ Faster convergence
✔ Better exploration + exploitation
✔ Robust hyperparameter stability
✔ Higher accuracy with less tuning
✔ Works well with noisy tunnel datasets

📂 Project Folder Structure
Underground Metro Tunnel Crack Detection/
│
├── 7261049/
│   └── 16. Tzika et al..pdf       # Source Dataset (PDF)
│
├── extracted_frames/              # Auto-generated images
│
├── ais_gwo_metrosense_tzika_model.h5        # Saved model
├── ais_gwo_metrosense_tzika_scaler.pkl      # Saved scaler
├── ais_gwo_metrosense_tzika_config.yaml     # Best HP config
├── ais_gwo_metrosense_tzika_result.csv      # Final results
├── ais_gwo_metrosense_tzika_prediction.json # Prediction report
│
├── ais_gwo_accuracy.png
├── ais_gwo_loss.png
├── ais_gwo_prediction.png
├── ais_gwo_result.png
├── ais_gwo_comparison.png
├── ais_gwo_confusion_matrix.png
│
└── README.md

📊 Graphs Generated

MetroSense AIS+GWO automatically generates and displays the following graphs:

📈 1. Accuracy Curve (Train vs Val)

Shows how the model improves over epochs.

📉 2. Loss Curve (Train vs Val)

Indicates convergence behaviour.

🔮 3. Prediction Graph

Compares predicted labels vs actual labels (first 50 samples).

📊 4. Result Distribution Graph

Shows count of crack vs no-crack samples.


![Confusion Matrix Heatmap](ais_gwo_loss.png)


🔍 5. Actual vs Predicted Comparison Graph

Highlights model consistency.

🟥 6. Confusion Matrix

Displays classification performance.

All graphs are shown on screen + saved automatically.

🧠 Model Architecture

A lightweight CNN is used:

Conv2D → MaxPooling2D → Conv2D → MaxPooling2D

Flatten

Dense Layer (tuned by AIS+GWO)

Dropout Layer (optimized)

Output Layer (sigmoid)

Hyperparameters optimized:

Parameter	Range
Learning Rate	0.0001 – 0.005
Filters	16, 32, 48, 64
Dense Units	64, 128, 256, 512
Dropout	0.1 – 0.5
🤖 Hybrid AIS + GWO Optimization Flow
Step 1: Create initial population of random hyperparameters

6 candidate solutions are generated.

Step 2: Evaluate population

Each HP set is trained for 2 quick epochs.

Step 3: AIS Phase

Clone top 2

Mutate LR + dropout

Produce new elite candidates

Step 4: GWO Phase

Identify Alpha, Beta, Delta wolves

Update positions of all HPs

Converge towards best solution

Step 5: Clamp + sanitize HPs

Ensures no negative or invalid values (e.g., negative units).

Step 6: Train Final Model

Using the optimized best HPs for 10 epochs.

🛠 System Requirements
🖥 Python Required:

Python 3.9–3.12

TensorFlow 2.x

pdf2image

Poppler (Windows)

📦 Install Dependencies:
pip install tensorflow pdf2image pillow numpy matplotlib seaborn scikit-learn tqdm pyyaml

🔧 Install Poppler

Download from:
https://github.com/oschwartz10612/poppler-windows/releases/

Add this to PATH:

C:\Users\NXTWAVE\Downloads\Release-25.11.0-0\poppler-25.11.0\Library\bin

🧪 Running the Code

Run the hybrid AIS+GWO script:

python ais_gwo_metrosense.py


Or in Jupyter Notebook:

%run ais_gwo_metrosense.py

📁 Model Outputs

All outputs are saved with ais_gwo_ prefix, including:

✔ Model

ais_gwo_metrosense_tzika_model.h5

✔ Scaler

ais_gwo_metrosense_tzika_scaler.pkl

✔ Config

ais_gwo_metrosense_tzika_config.yaml

✔ Results

ais_gwo_metrosense_tzika_result.csv

✔ Predictions

ais_gwo_metrosense_tzika_prediction.json

✔ Graphs

ais_gwo_accuracy.png

ais_gwo_loss.png

ais_gwo_prediction.png

ais_gwo_result.png

ais_gwo_comparison.png

ais_gwo_confusion_matrix.png

🧩 Complete Workflow Summary

Extract images from PDF

Preprocess images → detect crack edges

Create train-test split

Run AIS+GWO optimization (Generations = 5)

Select best hyperparameter configuration

Train final CNN model

Generate predictions

Save prediction JSON, result CSV

Display and save all graphs

Export model + config + scaler

🚀 Future Improvements

Vision Transformer (ViT) for tunnel crack classification

BiLSTM vibration forecasting

PSO+GWO hybrid for deeper optimization

Full Streamlit dashboard

IoT firmware for ESP32-CAM + MPU6050

3D tunnel digital twin

🏁 Conclusion

MetroSense AIS+GWO is a robust tunnel crack detection pipeline capable of:

✔ Automated dataset extraction
✔ Strong AI-based hyperparameter tuning
✔ Accurate crack classification
✔ Engineering-ready outputs (graphs, CSV, JSON, model)
