# Image Classifier Project

![ML Framework](https://img.shields.io/pypi/v/torch?label=torch)
![Vision Framework](https://img.shields.io/badge/torchvision-v0.1.6-B31B1B?logo=pytorch)

## 1. Project Name
**Flower Species Image Classifier** - Image classifier to recognize different species of flowers

## 2. Brief Description
This project is a deep learning-based image classifier designed to recognize and classify different species of flowers. It leverages pre-trained convolutional neural networks (CNNs) like DenseNet121 and VGG19, fine-tuned on a flower dataset, to achieve high accuracy in species identification. The project includes both training and prediction scripts, making it suitable for end-to-end machine learning workflows.

## 3. Main Features
- **Flexible Model Architecture**: Supports both DenseNet121 and VGG19 as backbone models.
- **Customizable Training**: Adjustable hyperparameters such as learning rate, dropout, hidden units, and epochs.
- **GPU Support**: Utilizes CUDA for accelerated training and inference if available.
- **Comprehensive Data Augmentation**: Includes resizing, cropping, rotation, and normalization for robust training.
- **Prediction Script**: Classifies single images and returns top-K probable classes with confidence scores.
- **Model Saving and Loading**: Saves trained models as checkpoints for future use.

## 4. Prerequisites
Before running the project, ensure you have the following installed:
- **Python 3.8+**
- **PyTorch 2.6.0** (with torchvision)
- **Pillow** (for image processing)
- **NumPy**
- **argparse** (for command-line arguments)
- **json** (for label mapping)

All dependencies are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

## 5. Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/CarlosYazid/Image-Classifier-Project.git
   cd Image-Classifier-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset (ensure it follows the structure below):
   ```
   data_dir/
   ├── train/
   ├── valid/
   └── test/
   ```

## 6. Usage
### Training the Model
Run the training script with optional arguments:
```bash
python train.py --data_dir ./flowers --save_dir ./models --arch densenet121 --gpu --epochs 20
```
**Key Arguments**:
- `--data_dir`: Path to the dataset directory.
- `--save_dir`: Directory to save the trained model.
- `--arch`: Model architecture (`densenet121` or `vgg19`).
- `--gpu`: Enable GPU acceleration.
- `--epochs`: Number of training epochs.

### Making Predictions
Use the trained model to classify an image:
```bash
python predict.py ./test_image.jpg ./models/checkpoint.pth --category_names cat_to_name.json --top_k 3 --gpu
```
**Key Arguments**:
- `image_path`: Path to the image to classify.
- `save_path`: Path to the trained model checkpoint.
- `--category_names`: JSON file mapping class indices to flower names.
- `--top_k`: Number of top predictions to display.
- `--gpu`: Use GPU for inference.

## 7. Examples
### Example 1: Training
```bash
python train.py --data_dir ./flowers --arch vgg19 --learning_rate 0.0005 --hidden_units 512 --epochs 15 --gpu
```
This trains a VGG19 model with a custom learning rate, hidden layer size, and 15 epochs using GPU.

### Example 2: Prediction
```bash
python predict.py ./rose.jpg ./models/checkpoint.pth --category_names cat_to_name.json --top_k 5
```
Output:
```
1. rose (98.7%)
2. hibiscus (0.8%)
3. tulip (0.3%)
4. sunflower (0.1%)
5. daisy (0.1%)
```

## 8. Project Structure
```
Image-Classifier-Project/
├── .gitignore
├── predict.py               # Script for making predictions
├── train.py                 # Script for training the model
├── cat_to_name.json         # Mapping of class indices to flower names
└── requirements.txt         # Python dependencies
```

## 9. API Reference
### `train.py`
- **`train_model`**: Trains the model using specified hyperparameters.
- **`accuracy`**: Evaluates model accuracy on the test set.
- **`save_model`**: Saves the trained model as a checkpoint.

### `predict.py`
- **`process_image`**: Preprocesses an image for the model.
- **`load_model`**: Loads a saved model checkpoint.
- **`predict`**: Predicts the class of an input image.
- **`display_prediction`**: Prints the top-K predictions.

## 10. How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a clear description of changes.
4. Ensure all tests pass and adhere to the project's coding standards.

## 11. Troubleshooting
- **CUDA Out of Memory**: Reduce batch size or use a smaller model.
- **Invalid Image Path**: Ensure the path is correct and the image exists.
- **JSON Decode Error**: Verify `cat_to_name.json` is valid JSON.

## 12. Changelog
### v1.0.0 (Initial Release)
- Added training and prediction scripts.
- Supported DenseNet121 and VGG19 architectures.
- Included GPU acceleration.

## 13. License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 14. Contact

For questions or support, please contact:

- **Project Maintainer**: Carlos Yazid
- **Email**: contact@carlospadilla.co
- **GitHub Issues**: [Issues](https://github.com/CarlosYazid/Image-Classifier-Project/issues)