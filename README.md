# image-colorization

# Deep Learning Image Colorization: Encoder-Decoder Architecture

## Introduction
One Convolutional Neural Network-based encoder-decoder model faithfully colorizes black and white images without any human assistance, while preserving the natural tonality. We made our best efforts to produce more aesthetically pleasing output than many existing solutions based on regression or classification. Moreover, it’s much more time-efficient than the traditional method of colorization by human assistance. Without the need to preprocess the data for feature extraction, the model can work in conjunction with a VGG16, a domain adaptation model widely used for computer vision applications, to learn image features. We have explored numerous cutting-edge concepts, including color spaces, Convolutional Neural Network architecture, Encoder-Decoder type domain adaptation, Transfer learning, regularization, and model evaluation, among others.


## 1. Project Overview
This project presents a Convolutional Neural Network (CNN) based encoder-decoder architecture designed to automatically colorize black and white (B&W) images without human intervention. The goal is to enhance the visual utility of monochromatic images—especially in fields like medical or general aesthetics—by predicting the tone and hue information lost in the B&W representation.

The model accepts a B&W image as input and outputs a colorized version, leveraging deep convolutional layers to learn and reproduce natural color features.

## 2. Methodology and Architecture
The system is built upon a **Deep Convolutional Neural Network** following a classic **Encoder-Decoder** structure.

* **Color Space:** We utilize the **LAB color space**, which allocates one channel (L) for lightness (B&W) and two channels (A and B) for color (blue-yellow and green-red). This simplifies the problem, as the model only needs to predict the two color channels (A and B) given the lightness channel (L).
* **Encoder (Feature Extraction):** To save computational resources and leverage powerful feature learning, the encoder integrates the first 18 layers of a pre-trained **VGG16** model. This is a form of transfer learning, which aids in domain adaptation by exploiting learned spatial hierarchies.
* **Decoder (Color Prediction):** The decoder side uses upsampling and de-convolutional layers to reconstruct the image's spatial dimensions while predicting the A and B channels. **Residual connections** are implemented to merge intermediate outputs from the encoder with the decoding portion, ensuring spatial information is preserved.
* **Implementation:** The entire architecture was built using **Keras** (in Python), with preprocessing handled by **OpenCV** and matrix manipulation optimized via **NumPy**.

## 3. Data Sourcing and Preparation
High-quality, complete, and varied data were crucial for training the model's color generalization capabilities. We sourced and aggregated several large datasets, totaling over 24,000 images:

| Dataset Name | Size (Approx.) | Primary Category Coverage |
| :--- | :--- | :--- |
| Google's Landscape and Nature | 2,688 | Landscape and Nature |
| Google's Human Gesture recognition | 4,000 | Human (Skin/Clothing variety) |
| MIT CVCL Urban and Natural Scene | 4,319 | Landscape and Natural Scene |
| Wallpapers from web | 15,000 | Landscape and Nature |

### Preprocessing Steps:
1. **Color Space Conversion:** All images were immediately converted to the **LAB color space**.
2. **Sizing:** Images were resized to **224x224x3**, matching the required input dimensions for the VGG16 base.
3. **Data Extraction:** The L channel (B&W) was extracted as the training input, and the A/B channels were used as the target output (ground truth).
4. **Normalization:** Pixel values, which typically range from -127 to 127 in LAB space, were normalized by dividing them by 128.

## 4. Performance and Results
* **Training:** The model was trained for 1,500 epochs using the Mean Squared Error (MSE) loss function.
* **Regularization:** **L1 and L2 regularization** were critical additions to the training process to combat overfitting, successfully reducing the muted/sepia-toned outputs and increasing color vibrancy.
* **Accuracy:** Testing accuracy reached **83%**, with a validation accuracy of **76%**.
* **Strengths:** The model performed remarkably well in coloring predictable scenes, such as **foliage, skies, and grass**.
* **Challenges:** The model struggled with multi-colored objects or subjects requiring high color precision (human skin and clothing), often producing a generalized, subdued sepia tone due to the inherent complexity of mapping a single L-channel value to a wide range of possible A/B color pairs.

## 5. Future Work
Future improvements should focus on enhancing color vibrancy and precision:
* Investigating alternative Encoder-Decoder implementations (non-CNN types).
* Exploring different fusion methods within the CNN architecture to separately optimize feature extraction and colorization.
* Utilizing post-processing schemes such as total variation minimization and conditional random fields to improve the dynamic range and reduce noise blobs.
