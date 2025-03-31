# Audio Deepfake Detection

## Overview
This project focuses on detecting AI-generated speech using deep learning models. The implementation uses the **RawNet2** architecture for classification and is trained on the **ASVspoof2019 LA dataset**. The system aims to enhance robustness against manipulated audio content in real-world applications.

## Features
- **Forgery Detection**: Identifies AI-generated speech vs. real human speech.
- **RawNet2 Model**: End-to-end deep learning approach using raw waveform inputs.
- **Dataset Support**: Compatible with ASVspoof and other deepfake datasets.
- **Fine-tuning Capabilities**: Model can be adapted for new datasets.
- **Performance Metrics**: Evaluates results using EER and AUC scores.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/audio-deepfake-detection.git
   cd audio-deepfake-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Training
To train the model:
```bash
python train.py --epochs 5 --batch_size 16
```

## Evaluation
To evaluate the trained model:
```bash
python evaluate.py --model rawnet2_deepfake_detector.pth
```

## Results
- **Equal Error Rate (EER):** ~2.0%
- **AUC Score:** 0.98

## Future Improvements
- Improve generalization using domain adaptation.
- Optimize for real-time detection.
- Implement hybrid approaches for better accuracy.

https://drive.google.com/file/d/1EiXfj3fvzUa1Y3r0iNuTKJusJb7DW-fJ/view?usp=sharing
