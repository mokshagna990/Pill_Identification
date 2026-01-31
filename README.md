# Pill Identification System

A deep learning-based system for identifying pharmaceutical pills using computer vision and neural networks.

## Project Overview

This project implements a pill identification system that can classify 20 different types of pills using deep learning models (MobileNetV2 and EfficientNet). The system consists of:

- **Backend**: Model training pipeline using TensorFlow/Keras
- **Frontend**: Django web application for pill identification

## Supported Pills

The system can identify the following 20 types of pills:

1. Amoxicillin 500 MG
2. Atomoxetine 25 MG
3. Calcitriol 0.00025 MG
4. Oseltamivir 45 MG
5. Ramipril 5 MG
6. Apixaban 2.5 MG
7. Aprepitant 80 MG
8. Benzonatate 100 MG
9. Carvedilol 3.125 MG
10. Celecoxib 200 MG
11. Duloxetine 30 MG
12. Eltrombopag 25 MG
13. Montelukast 10 MG
14. Mycophenolate Mofetil 250 MG
15. Pantoprazole 40 MG
16. Pitavastatin 1 MG
17. Prasugrel 10 MG
18. Saxagliptin 5 MG
19. Sitagliptin 50 MG
20. Tadalafil 5 MG

## Project Structure

```
Pill_identification/
├── BACKEND/
│   └── Pill_Identification.ipynb    # Model training notebook
├── DATASET/                          # Training images (organized by pill type)
├── FRONTEND/                         # Django web application
│   ├── manage.py
│   ├── new_project/                  # Django project settings
│   ├── new_app/                      # Main application
│   ├── templates/                    # HTML templates
│   ├── static/                       # CSS, JS, images
│   ├── Efficientnet.h5              # Trained EfficientNet model
│   ├── mobilenetv2_final.h5         # Trained MobileNetV2 model
│   ├── labels.txt                    # Class labels
│   └── pills_description.csv         # Pill information database
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Features

- **Dual Model Architecture**: Uses both MobileNetV2 and EfficientNet for robust predictions
- **Web Interface**: User-friendly Django web application
- **Image Upload**: Upload pill images for identification
- **Detailed Information**: Provides pill name, dosage, and description
- **High Accuracy**: Trained on 994 images with stratified train-test split

## Installation

### Prerequisites

- Python 3.7+
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. **Clone or download the project**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to the frontend directory**
   ```bash
   cd FRONTEND
   ```

5. **Run database migrations**
   ```bash
   python manage.py migrate
   ```

6. **Start the development server**
   ```bash
   python manage.py runserver
   ```

7. **Access the application**
   - Open your browser and go to: `http://127.0.0.1:8000/`

## Usage

### Web Application

1. Start the Django server (see installation steps above)
2. Navigate to the home page
3. Upload an image of a pill
4. Click "Identify"
5. View the prediction results with pill information

### Model Training

To retrain the models:

1. Open `BACKEND/Pill_Identification.ipynb` in Jupyter Notebook or Google Colab
2. Ensure your dataset is properly organized in the `DATASET` folder
3. Run all cells in the notebook
4. The trained models will be saved as `.h5` files

## Models

### MobileNetV2
- Lightweight model optimized for mobile and embedded devices
- Fast inference time
- Good accuracy-to-size ratio

### EfficientNet
- State-of-the-art convolutional neural network
- Better accuracy with efficient scaling
- Larger model size but superior performance

## Dataset

- **Total Images**: 994
- **Classes**: 20 different pill types
- **Split**: 80% training, 20% testing (stratified)
- **Augmentation**: Applied during training for better generalization

## Technical Details

- **Framework**: TensorFlow 2.x / Keras
- **Web Framework**: Django 3.x
- **Image Processing**: OpenCV, PIL
- **Data Handling**: NumPy, Pandas
- **Visualization**: Matplotlib

## Performance

The models achieve high accuracy on the test set. Specific metrics can be found in the training notebook.

## Future Improvements

- [ ] Add more pill types to the database
- [ ] Implement real-time camera capture
- [ ] Add multi-pill detection in single image
- [ ] Deploy to cloud platform
- [ ] Create mobile application
- [ ] Add user authentication
- [ ] Implement pill interaction warnings

## Troubleshooting

### Common Issues

1. **Model files not found**
   - Ensure `Efficientnet.h5` and `mobilenetv2_final.h5` are in the FRONTEND directory
   - If missing, retrain models using the notebook

2. **Import errors**
   - Verify all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Database errors**
   - Run migrations: `python manage.py migrate`
   - Delete `db.sqlite3` and run migrations again if needed

## License

This project is for educational purposes.

## Acknowledgments

- Dataset sourced from pharmaceutical pill images
- Built using TensorFlow and Django frameworks
- Inspired by medical AI applications

## Contact

For questions or contributions, please open an issue or submit a pull request.

---

**Note**: This system is for educational purposes only and should not be used as a substitute for professional medical advice.
