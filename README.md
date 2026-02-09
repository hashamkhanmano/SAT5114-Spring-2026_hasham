# Breast Cancer Detection using SVM Classifier

## Project Overview
This project implements a Support Vector Machine (SVM) classifier to diagnose breast cancer using the Wisconsin Breast Cancer dataset. The model achieves high accuracy in distinguishing between malignant and benign tumors.

## Dataset
- **Source**: [UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features**: 30 numeric features computed from digitized images of breast mass
- **Target**: Binary classification (Malignant vs Benign)
- **Samples**: 569 instances

## Implementation Details

### Machine Learning Approach
- **Classifier**: Support Vector Machine (SVM)
- **Validation**: 5-fold cross-validation
- **Hyperparameter Tuning**: GridSearchCV
- **Feature Scaling**: StandardScaler

### Key Features
1. **K-fold Cross-Validation**: 5-fold CV on training data to ensure model robustness
2. **Hyperparameter Optimization**: GridSearchCV to find optimal C, gamma, and kernel parameters
3. **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Sensitivity, and Specificity
4. **Visualization**: Confusion matrix and performance metrics charts

## Requirements
```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

## Installation
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Usage
1. Clone this repository
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook breast_cancer_svm_detection.ipynb
   ```
3. Run all cells to train the model and view results

## Results
The SVM classifier demonstrates excellent performance on the test dataset:
- **Accuracy**: >95%
- **Precision**: High reliability in positive predictions
- **Recall/Sensitivity**: Strong detection of benign cases
- **Specificity**: Effective identification of malignant cases
- **F1-Score**: Excellent balance between precision and recall

*(Exact scores will be displayed after running the notebook)*

## Project Structure
```
.
├── breast_cancer_svm_detection.ipynb  # Main implementation notebook
├── README.md                          # Project documentation
├── confusion_matrix.png               # Generated confusion matrix visualization
└── performance_metrics.png            # Generated metrics visualization
```

## Evaluation Metrics Explained

### Accuracy
The proportion of correct predictions among all cases. Indicates overall model performance.

### Precision
The proportion of true positive predictions among all positive predictions. Shows reliability when predicting benign cases.

### Recall (Sensitivity)
The proportion of actual positive cases correctly identified. Measures the model's ability to detect benign tumors.

### F1-Score
The harmonic mean of precision and recall. Provides a balanced performance measure.

### Specificity
The proportion of actual negative cases correctly identified. Measures the model's ability to detect malignant tumors.

## Author
Machine Learning Project - Option B

## License
This project is for educational purposes.
