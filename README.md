# Brain Tumor Classification Using Deep Learning(CNN)
## Project Overview
Brain tumors are abnormal growths of cells within the brain that can be life-threatening if not detected early. Accurate and timely diagnosis is crucial for effective treatment. This project focuses on building an automated system using deep learning to classify brain 
## MRI scans into four categories:
1.Glioma Tumor
2.Meningioma Tumor
3.Pituitary Tumor
4.No Tumor (healthy brain)

The model aims to assist radiologists and medical professionals by providing a reliable tool for early detection and classification of brain tumors.

# Why This Project?
Manual examination of MRI scans is time-consuming, prone to human error, and requires expert knowledge. Automating tumor classification can:
->Speed up diagnosis
->Improve accuracy and consistency
->Assist in treatment planning
->Provide accessibility where expert radiologists are scarce

## Dataset Description
$ The dataset consists of MRI images collected and organized into two main parts:
**Training Set: Used to train the deep learning model
**Testing Set: Used to evaluate model performance on unseen data
Each set contains subfolders corresponding to four classes of brain conditions:
1.glioma_tumor
2.meningioma_tumor
3.pituitary_tumor
4.no_tumor
Images are preprocessed by resizing to a fixed size of 150x150 pixels to standardize input for the CNN model.

# Methodology
## 1. Data Preprocessing
*Image Loading: Images are read from directories using OpenCV (cv2) and resized to 150x150 pixels.
*Label Encoding: Each tumor type is assigned an integer label (0 to 3) and converted into one-hot vectors to suit multi-class classification.
*Normalization: Pixel values are scaled to the range [0,1] for faster convergence.

## 2. Model Architecture
*A Convolutional Neural Network (CNN) is designed to learn features from MRI images:
$ Convolutional Layers (Conv2D): Extract spatial features like edges, textures, and shapes.
$ Activation Function (ReLU): Introduces non-linearity to learn complex patterns.
$ MaxPooling Layers: Reduce spatial dimensions to lower computation and extract dominant features.
$ Dropout Layers: Randomly deactivate neurons during training to prevent overfitting.
$ Fully Connected (Dense) Layers: Learn high-level features and perform classification.
$ Output Layer: Uses softmax activation to provide probabilities across the 4 tumor classes.

## 3. Training
The model is trained for 20 epochs with a batch of images.
10% of the training data is set aside for validation to monitor the model’s ability to generalize.
Accuracy and loss are tracked for both training and validation.

## 4. Evaluation
Training and validation accuracy/loss curves are plotted to assess learning.
Early detection of overfitting or underfitting helps in model tuning.

## 5. Prediction
The trained model can predict the tumor class of new MRI scans.
Input images are preprocessed in the same way before prediction.
The model outputs the predicted tumor category with the highest probability.

# Results
* The CNN model achieves high accuracy in distinguishing between tumor types.
* Visualization of accuracy and loss over epochs indicates effective learning.
* The model generalizes well on validation data, showing potential for real-world application.

## How to Use This Project
Running the Model
Clone the repository:

bash
```
git clone https://github.com/sowmya13531/Brain-Tumor-Classification-Using-Deep-Learning.git
cd Brain-Tumor-Classification-Using-Deep-Learning
```

## Install required libraries:
bash
```
pip install -r requirements.txt
```
Run the Jupyter Notebook or Python scripts to train or test the model.

## Predicting on New Images
Load the image, resize it to 150x150, normalize, and convert to a batch tensor.
Pass it to the model for prediction.
Interpret the output to get the tumor type.

## Technologies Used
Python 3
TensorFlow/Keras for deep learning
OpenCV for image processing
NumPy for numerical operations
Matplotlib for plotting and visualization
scikit-learn for preprocessing utilities

# Future Work
> Incorporate more diverse datasets to improve model robustness.
> Experiment with advanced architectures like ResNet, EfficientNet.
> Implement real-time tumor detection on video or 3D MRI scans.
> Build a user-friendly web or mobile app interface for practical deployment.

Contact
For questions, suggestions, or collaborations, please reach out at:
[Sowmya Kanithi]
Email: kanithisowmya2005@gmail.com

⭐ If this project helped you, please consider giving it a star! 
