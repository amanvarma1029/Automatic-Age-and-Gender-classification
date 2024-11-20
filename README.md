
# Automatic Age and Gender Classification

This project focuses on accurately classifying the age and gender of individuals from facial images, leveraging deep learning and convolutional neural networks (CNNs). It combines advanced feature extraction techniques with pre-trained models to enhance performance, particularly on real-world images. 

## Features

- **Deep Learning Models**: Utilizes `age_net.caffemodel` and `gender_net.caffemodel` for robust classification.
- **Improved Accuracy**: Integrates handcrafted features and CNNs for better prediction results, especially when data is limited.
- **Multi-Age Group Accuracy**: Measures exact age-group classification and accounts for adjacent group predictions to handle task uncertainty.

## System Architecture

The project includes a simple yet effective convolutional net architecture for learning representations, enhancing classification performance over traditional methods like:

- Gaussian Mixture Models (GMM)
- Support Vector Machines (SVM)
- AdaBoost

## Advantages

- Better handling of real-world image variability compared to traditional methods.
- Incorporates viewpoint-invariant techniques for improved accuracy.
- Addresses age-group overlap for realistic classification scenarios.

## Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/age-gender-classification.git
   cd age-gender-classification
   ```

2. **Download Pre-trained Models**:
   - [Gender Net](https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0)
   - [Age Net](https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=0)
   Place the downloaded files in the `models/` directory.

3. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Code**:
   For Python:
   ```bash
   python AgeGender.py --input <input_file>
   ```
   Leave `<input_file>` blank to use the webcam.

   For C++:
   ```bash
   cmake .
   make
   ./AgeGender <input_file>
   ```

### Software
- **Language**: Python
- **Operating Systems**: Windows,Mac

### Hardware

- Storage: Minimum 512 MB free space


This project demonstrates the integration of handcrafted features with deep learning to achieve superior age and gender classification. By enabling CNNs to focus on meaningful features, it achieves improved accuracy and performance on real-world datasets.

---

Feel free to adjust the file paths or links as necessary!
