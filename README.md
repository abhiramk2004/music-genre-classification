
# Music Genre Classification

This project focuses on classifying music genres from audio clips longer than 3 seconds, leveraging deep learning and signal processing techniques. It is built upon the **GTZAN dataset**, a well-known benchmark for music genre classification.

### Objective

To design and deploy a robust audio classification model capable of identifying musical genres from longer audio clips with high accuracy. The longer the input audio, the more reliable the classification—thanks to temporal averaging and ensemble prediction.

## Dataset

* **GTZAN** music genre dataset
* 10 genres: `blues`, `classical`, `country`, `disco`, `hiphop`, `jazz`, `metal`, `pop`, `reggae`, `rock`

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed to process 5-channel feature representations of audio signals, transformed into 2D image-like data.

### Audio-to-Image Transformation

To leverage CNNs effectively, audio waveforms are converted to multi-channel spectro-temporal representations:

* **MFCC** (Mel-Frequency Cepstral Coefficients)
* **Delta MFCC** (First-order temporal derivative)
* **Delta-Delta MFCC** (Second-order temporal derivative)
* **Tempo-synced MFCC** (beat-aware dynamics)
* **Chroma** (harmonic pitch classes)

These 5 channels are stacked to create a `(40, 130, 5)` tensor, encoding rhythmic, spectral, and harmonic content.

### Architecture Breakdown

#### Block 1

```
Conv2d(5 → 32) → ReLU → BatchNorm2d → MaxPool2d(2) → Dropout(0.3)
```

* Extracts 32 local features from 5-channel input
* Reduces spatial dimension from `40×130` to `20×65`

#### Block 2

```
Conv2d(32 → 64) → ReLU → BatchNorm2d → MaxPool2d(2) → Dropout(0.3)
```

* Expands features to 64 maps
* Output size: `10×32`

#### Adaptive Pooling

```
AdaptiveAvgPool2d((1, 1)) → Flatten → Linear(64 → 128) → ReLU → BatchNorm1d → Dropout(0.3)
Linear(128 → 10) → LogSoftmax
```

* Transforms spatial features to class logits

### Output

* **Shape:** `(batch_size, 10)`
* **Values:** Log-probabilities for each genre class

## Design Highlights
 **BatchNorm**          Accelerates and stabilizes training                
 **Dropout (0.3)**      Regularizes model and prevents overfitting         
 **AdaptiveAvgPool2d**  Allows fixed-size output regardless of input scale 


## Training Pipeline

### Tools and Techniques

* **Mixed-Precision Training:** via `torch.cuda.amp.GradScaler` for efficient GPU utilization
* **Loss Function:** `CrossEntropyLoss` for multi-class classification
* **Optimizer:** `Adam` with weight decay (`1e-4`) for better generalization
* **Metric:** `torchmetrics.Accuracy` for consistent evaluation

### Training Loop

Per Epoch:

* Iterate through training batches
* Apply AMP scaling for backpropagation
* Track and average loss, update accuracy

Validation:

* Run model in `no_grad` mode
* Accumulate validation loss and accuracy metrics

---

## Evaluation

* **Test Accuracy:** Computed using withheld test data
* **Confusion Matrix:** Used to analyze class-specific misclassifications, especially among challenging pairs like:

  * `hiphop` vs `reggae`
  * `disco` vs `pop`
  * `classical` vs `jazz`
  * `metal` vs `rock`

---

## Web Interface (Flask)

A lightweight Flask web server provides a user interface for real-time inference.

### Upload Flow

* Accepts audio files via a form
* Splits input into 3-second chunks
* Classifies each chunk individually
* Uses majority voting to determine the final genre

### Features

* Easy drag-and-drop UI
* Compatible with both CPU and GPU environments
* Fast and scalable prediction pipeline

## Compatibility

The model is designed to automatically detect and utilize available hardware:
## To run the app
```
   git clone https://github.com/abhiramk2004/music-genre-classification/
   cd music-genre-classification
   python -m venv venv
   source/venv/Script/activate
   pip install -r requirement.txt
   python mainthing.py
```
## To run the model training session
```
   git clone https://github.com/abhiramk2004/music-genre-classification/
   cd music-genre-classification
   python -m venv venv
   source/venv/Script/activate
   pip install -r requirement.txt
   python mainthing.py
```
## Accuracy
* Train accuracy:`86.42`
* Validation accuracy:`85.62`
* Test accuracy:`85.90`


## Some Insights
* I did training with genre specific augmentation, but made a class more missclassified
* Adding the original one generalized the model, made the same audio without augmentation less misclassified
* At first i made the model using keras, after i shifted the whole model and training process to pytorch, the shifting helped me to use cuda more efficiently
* Understood about the need of three:train,validation,test dataset
* Different model parameters are saved as a checkpoint 
