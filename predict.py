import numpy as np
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image
import json
from collections import OrderedDict
#from time import time
import argparse


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')

    # Paths
    parser.add_argument('image_path', type = str, 
                    help = 'Provide the path to a singe image (required)')
    parser.add_argument('save_path', type = str, 
                    help = 'Provide the path to the file of the trained model (required)')

    # Model parameters
    parser.add_argument('--category_names', type = str,
                    help = 'Use a mapping of categories to real names')
    parser.add_argument('--top_k', type = int, default = 5,
                    help = 'Return top K most likely classes. Default value is 5')
    
    # Image transformations
    parser.add_argument('--image_size', type = int, default = 256,
                    help = 'Image size, default value is 256')
    parser.add_argument('--rescale', type = int, default = 224,
                    help = 'Rescale image, default value is 224')
    
    # GPU
    parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

    args_in = parser.parse_args()

    # Image transformations
    size = args_in.image_size
    resize_min = args_in.rescale
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    transforms_ = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(resize_min),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    with open(args_in.category_names, 'r') as f:
        classes = json.load(f)
    class_long = len(classes)

    def process_image(image, transform = transforms_):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        img = Image.open(image)
        return np.array(transform(img))

    def load_model(filepath = "checkpoint.pth", pre_trained_model = None, classifier = None):
    
        model_info = torch.load(filepath)
    
        if pre_trained_model is None:
            pre_trained_model = models.densenet121(weights= models.densenet.DenseNet121_Weights.IMAGENET1K_V1) if model_info['transfer_model'] == 'densenet121' else models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1)
    
        if classifier is None:
            pre_trained_model.classifier = model_info['classifier']
        else:
            pre_trained_model.classifier = classifier
        
        pre_trained_model.load_state_dict(model_info['state_dict'])
    
        pre_trained_model.class_to_idx = model_info['class_to_idx']
    
        return pre_trained_model, model_info


    def predict(image_path, model_path = "checkpoint.pth", topk = 5):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        with torch.no_grad():
        
            image = process_image(image_path)
            image = torch.from_numpy(image)
        
            image.unsqueeze_(0)
        
            image = image.float()
            model, _ = load_model(model_path)
            outputs = model(image)
            probs, classes = torch.exp(outputs).topk(topk)
        
            return zip(probs[0].tolist(), classes[0].add(1).tolist())
        
        def display_prediction(results, categories):
            i = 0
            for prob, class_ in results:
                i += 1
                prob = str(round(prob,4) * 100.) + '%'
            
                if (categories):
                    class_ = categories[str(class_)]
                else:
                    class_ = f' class {str(class_)}'
                print(f"{i}.{class_} ({prob})")
                
        result = predict(args_in.image_path, args_in.save_path, args_in.top_k)
        display_prediction(result, classes)
