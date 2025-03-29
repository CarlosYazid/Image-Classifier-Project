import numpy as np
import torch
import json
import argparse
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
#from time import time

def train_model(model, classifier = None, criterion = None, optimizer = None,
                trainloader = None, validloader = None, device = None, epochs = 10,
               valid_frecuency = 5):
    
    if (classifier is not None and
        criterion is not None and
        optimizer is not None and
        trainloader is not None and
        validloader is not None):
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = classifier
    
    
        model.to(device)
    
        running_loss = 0
        steps = 0

        print("Training...")
        
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps += 1
            
                #Prepare enviroment
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                #Train
                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
          
                if steps % valid_frecuency == 0:
                
                    #Test
                    test_loss = 0
                    accuracy = 0
                    model.eval()
                
                    with torch.no_grad():
                        for inputs, labels in validloader:
                        
                            #Prepare enviroment
                            inputs, labels = inputs.to(device), labels.to(device)
                        
                            #Evaluate
                            logps = model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()
                        
                            # Calculate accuracy
                            ps = torch.exp(logps)
                            _ , top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                        #Logging
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/(valid_frecuency)}.. "
                          f"Test loss: {test_loss/len(validloader)}.. "
                          f"Test accuracy: {accuracy/len(validloader)}")
                    #Reset
                    running_loss = 0
                    model.train()

def accuracy(model, testloader, device = None):
    
    #Define device
    device = torch.device("cpu") if device is None else device
    accuracy = 0
    total = 0
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            _ , predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            
            del images, labels, outputs, predicted
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
    print(f'Accuracy of the network : {100 * accuracy / total}%')
    return accuracy / total 

def save_model(model, pre_trained_model = None, dataset = None, input_size = None,
              output_size = None, path = "./" ,filename = "checkpoint", device = None):
    
    if dataset is not None:
        model.class_to_idx = dataset.class_to_idx
    device = torch.device("cpu") if device is None else device
    model.to(device)
    
    checkpoint = {
        'features': model.features,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict
    } if (pre_trained_model is None or
          input_size is None or output_size is None) else {
        'transfer_model' : pre_trained_model,
        'input_size': input_size,
        'output_size': output_size,
        'features': model.features,
        'classifier': model.classifier,
        'class_to_idx': model.class_to_idx,
        'model_state_dict': model.state_dict(),
        'opt_state_dict': optimizer.state_dict
    } 
    
    torch.save(checkpoint, f'{path}{filename}.pth')
    
    print(f"Saved as: {filename}.pth")
    
    return checkpoint


if __name__ == "__main__":
    
    # Argument Parser 
    parser = argparse.ArgumentParser(description='Train a neural network to classify images')
    
    # Data directory
    parser.add_argument('--data_dir', type = str, 
                    help = 'Provide the data directory, mandatory')
    parser.add_argument('--save_dir', type = str, default = './',
                    help = 'Provide the save directory')
    parser.add_argument('--category_names', type = str, default='cat_to_name.json',
                    help = 'Provide the class labels file in json, mandatory')

    # Image Transformations
    parser.add_argument('--image_size', type = int, default = 256,
                    help = 'Image size, default value is 256')

    parser.add_argument('--batch_size', type = int, default = 64,
                    help = 'Batch size, default value is 64')

    parser.add_argument('--rescale', type = int, default = 224,
                    help = 'Rescale image, default value is 224')

    parser.add_argument('--rotation', type = int, default = 30,
                    help = 'Rotation angle, default value is 30')

    # Model architecture
    parser.add_argument('--arch', type = str, default = 'vgg19',
                    help = 'densenet121 or vgg19')

    # Model hyperparameters
    parser.add_argument('--learning_rate', type = float, default = 0.001,
                    help = 'Learning rate, default value 0.001')
    parser.add_argument('--hidden_units', type = int, default = 0,
                    help = 'Number of hidden units, default value is 0')
    parser.add_argument('--dropout', type = float, default = 0.2,
                    help = 'Dropout, default value 0.2')
    parser.add_argument('--epochs', type = int, default = 10,
                    help = 'Number of epochs, default value is 10')
    parser.add_argument('--print_every', type = int, default = 5,
                    help = 'Print every, default value is 5')

    # GPU
    parser.add_argument('--gpu', action='store_true',
                    help = "Add to activate CUDA")

    args_in = parser.parse_args()

    if args_in.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Data loading
    data_dir  = args_in.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    # Image transformations
    size = args_in.image_size
    resize_min = args_in.rescale
    means, stds = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    rotation = args_in.rotation
    batch_size = args_in.batch_size

    data_train_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(resize_min),
        transforms.RandomRotation(rotation),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    data_test_valid_transforms = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(resize_min),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    print("Data loading...")

    # Data loading
    train_dataset = datasets.ImageFolder(train_dir, transform = data_train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = data_test_valid_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = data_test_valid_transforms)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)

    with open(args_in.category_names, 'r') as f:
        classes = json.load(f)

    print("Data loaded\n\nBuilding Model...")

    class_long = len(classes)

    # Model
    model, size_output = models.densenet121(weights= models.densenet.DenseNet121_Weights.IMAGENET1K_V1), (16,1024) if args_in.arch == 'densenet121' else models.vgg19(weights=models.vgg.VGG19_Weights.IMAGENET1K_V1), (32,25088)
  
    size_input_layer = int((size_output[1] - class_long) * (1/2)) if args_in.hidden_units == 0 else args_in.hidden_units
    p = args_in.dropout

    # Classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args_in.learning_rate)

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(size_output[1], size_input_layer)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p)),
        ('fc2', nn.Linear(size_input_layer, class_long)),
        ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    print("Model built\n\n")
    
    train_model(model, classifier, criterion,
                optimizer, trainloader, validloader,
                device, args_in.epochs, args_in.print_every)
    
    print("Training completed\n\nTesting...")
    
    accuracy(model, testloader, device)
    
    print("Testing completed\n\nSaving model...")
    
    save_model(model, args_in.arch , size_output[1],
               class_long, path = args_in.save_dir, device = device) 
    
    print("Model saved") 