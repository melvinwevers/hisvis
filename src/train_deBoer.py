from __future__ import print_function, division

import argparse

import math
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
from timeit import default_timer as timer

import torch
from torch import cuda
import torch.optim as optim
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler

from torchsummary import summary
import torchvision.models as models

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from make_split import img_train_test_split


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_on_gpu = cuda.is_available()
checkpoint_path = 'resnet50-deboer.pth' #todo: move somewhere else
save_file_name = 'resnet50-deboer.pth' #todo: fix this


def get_arguments():
    parser = argparse.ArgumentParser(description='De Boer FineTuning')

    parser.add_argument(
        '--source', 
        default='../src/deboer',
        help='which dataset to split')

    parser.add_argument(
        '--output', 
        default='../src/output/',
        help='output folder')

    parser.add_argument(
        '--fixed_split', 
        type=int, default=0, 
        help='fixed number of test samples')

    parser.add_argument(
        '--min_files', 
        type=int, 
        default=20, 
        help='minimum number of training samples')


    parser.add_argument(
        '--data_dir', 
        default='../src/output/train/', 
        help='path to dataset'
        )
    parser.add_argument(
        '--test_data_dir', 
        default='../src/output/test/', 
        help='path to dataset'
        )
    parser.add_argument(
        '--batch_size', 
        default=128, 
        type=int,
        help='batch size (default: 64)'
        )
    parser.add_argument(
        '--seed', 
        default=666, 
        type=int,
        help='seed for initializing training.'
        )
    parser.add_argument(
        '--max_epochs_stop', 
        default=5,
        type=int,
        help='Early stop after this many epochs.'
        )
    parser.add_argument(
        '--folds', 
        default=10,
        type=int,
        help='Number of cross validation folds.'
        )
    parser.add_argument(
        '--save_folder', 
        default='weights/', type=str,
        help='path to save model. '
        )
    parser.add_argument(
        '--arch', 
        default='resnet50-places', type=str,
        help='pre-trained initialization model.'
        )
    parser.add_argument(
        '--workers', 
        default=8, 
        type=int,
        help='number of data loading workers (default: 8)'
        )
    parser.add_argument(
        '--feature_extract', 
        action='store_true', 
        help='use pre-trained model'
        )
    parser.add_argument(
        '--cross_val', 
        action='store_true', 
        help='use cross-validation'
        )
    parser.add_argument(
        '--predictions', 
        action='store_true', 
        help='export prediction file'
        )
    parser.add_argument(
        '--balanced', 
        action='store_true', 
        help='weight balace training set'
        )
    parser.add_argument(
        '--shuffle', 
        action='store_true', 
        help='shuffle dataloaders'
        )
    parser.add_argument(
        '--resume', 
        default='', 
        type=str, 
        metavar='PATH',
        help='path to latest checkpoint (default: none)'
        )
    parser.add_argument(
        '--n_epochs', 
        default=100, 
        type=int,
        help='number of total epochs to run'
        )
    parser.add_argument(
        '--valid_size', 
        default=0.2, 
        type=float,
        help='size of validation dataset'
        )
    parser.add_argument(
        "--cuda", 
        action="store_true", 
        help="Use Cuda."
        )
    parser.add_argument(
        '--print_every', 
        default=1, 
        type=float,
        help='determines how often to print progress'
        )
    parser.add_argument(
        '--evaluate', 
        dest='evaluate', 
        action='store_true',
        help='evaluate model on test set'
        )
    
    args = parser.parse_args()
    return args



def main():
    args = get_arguments()

    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    
    if not os.path.exists(args.output):
        img_train_test_split(args)

    results = []
    if args.cross_val:
        for i in range(args.folds):
            print('Run: {}'.format(i))
        
            main_process(args)
            results.append(pd.read_csv('class_accuracy.csv'))
        overall_results = pd.concat(results)
        # overall_results = overal_results.groupby('class').mean()
        overall_results.to_csv('cross_valid_results.csv', index=False)
    else:
        main_process(args)



def main_process(args):
    '''
    This function runs the main training process
    '''
    print('initializing datasets and dataloaders....')
    # load data transforms   
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.RandomRotation(20, resample=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),
            random_erase(.5, [0.2, 0.04], 0.3, 20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # load dataset
    train_dataset = datasets.ImageFolder(
        args.data_dir, 
        transform=data_transforms['train']
        )

    valid_dataset = datasets.ImageFolder(
        args.data_dir, 
        data_transforms['valid']
        )

    if args.evaluate:
        test_dataset = datasets.ImageFolder(
            args.test_data_dir, 
            data_transforms['test']
            )

    class_names = train_dataset.classes

    # Prepare validation set
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(args.valid_size * num_train))

    if args.shuffle:
        print('Shuffling data')
        np.random.seed(args.seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # initialize data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler, 
        num_workers=args.workers,
        pin_memory=True
        )

    if args.balanced:
        print('Using balanced training set')
        weights = make_weights_for_balanced_classes(
            train_dataset.imgs, 
            len(train_dataset.classes))      
        
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights), replacement=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=weighted_sampler, num_workers=args.workers)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        sampler=valid_sampler, 
        num_workers=args.workers,
        pin_memory=True
        )
        
    if args.evaluate:
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            sampler=valid_sampler, 
            num_workers=args.workers,
            pin_memory=True
            )
    

    # initialize model
    model = initialize_model(args.arch, args.feature_extract, class_names)
    model = remap_classes(model, train_dataset)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), amsgrad=True)


    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting Training from Scratch.\n')
    
    if args.resume:
        print('continue training ...')
        model.load_state_dict(torch.load(args.resume))

    if args.evaluate:
        class_accuracy(model, test_dataset, test_loader, top=(1,5))
        return

    epochs_no_improve = 0
    best_val_score = -np.Inf
    min_val_loss = np.Inf
    best_state_dict = None

    overall_start = timer()
    print('Training has commenced!')

    for epoch in range(args.n_epochs):
        # training step
        train_model(train_loader, model, criterion, optimizer, epoch, args)
        
        # validation step
        acc1, acc5, val_loss = validate_model(valid_loader, model, criterion, epoch, args)
        print("min_val_loss: {}".format(min_val_loss))
        print("val_loss: {}".format(val_loss))

        if val_loss < min_val_loss:
            best_val_score = acc1
            min_val_loss = val_loss
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
            best_epoch = epoch
            #torch.save(model.state_dict(), save_file_name) 
        else:
            epochs_no_improve += 1
            print('no improvement for {} Epochs!'.format(epochs_no_improve))
            
            # Trigger early stopping
            if epochs_no_improve == args.max_epochs_stop:
                if best_state_dict is not None:
                    model.load_state_dict(best_state_dict)
                    model.optimizer = optimizer
                    # calculate validation accuracy per class
                    class_accuracy(model, valid_idx, valid_loader, topk=(1, 5)) 
                    save_checkpoint(model, checkpoint_path)
                    if args.predictions:
                        make_predictions(model, valid_loader)
                    
                    print(f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with top1_acc: {best_val_score:.2f}  and top5_acc: {acc5:.2f}')
                    break
        total_time = timer() - overall_start
        print(f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.')


def train_model(train_loader, model, criterion, optimizer, epoch, args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    model.train()
    start = timer()

    for i, (data, target) in enumerate(train_loader):
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
  
            # make predictions
            output = model(data)
            loss = criterion(output, target)

            # measure training accuracy
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

            # Clear gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track training progress
            print(f'Epoch: {epoch}\t{100 * (i + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                        end='\r')

    if (epoch + 1) % args.print_every == 0:
        print('\nTraining progress')
        print(f'Loss: {losses.val:.4f} ({losses.avg:.4f})')
        print(f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
        print(f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')


def validate_model(valid_loader, model, criterion, epoch, args):
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()

    with torch.no_grad():
        model.eval()

        for ii, (data, target) in enumerate(valid_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = model(data)

            loss = criterion(output, target)

            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))

            # Print validation results
        if (epoch + 1) % args.print_every == 0:
            print('Validation progress')
            print(f'Loss: {losses.val:.4f} ({losses.avg:.4f})')
            print(f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
            print(f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')
    
    return top1.avg, top5.avg, losses.avg



def initialize_model(arch, feature_extract, class_names):
    model = None

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        if feature_extract:
            set_parameter_requires_grad(model)

        n_inputs = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, len(class_names)), nn.LogSoftmax(dim=1))

    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
        if feature_extract:
            set_parameter_requires_grad(model)

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, len(class_names)), nn.LogSoftmax(dim=1))

    elif arch == 'resnet50-places':
        arch2 = 'resnet50'
        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch2
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)
        model = models.__dict__[arch2](num_classes=365)
        checkpoint = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k, 'module.', ''): v for k,
                      v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        if feature_extract:
            set_parameter_requires_grad(model)

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(256, len(class_names)), nn.LogSoftmax(dim=1))

    else:
        print("invalid model name, exiting")
        exit()

    model = model.to(device)

    print("Params to learn:")
    params_to_update = model.parameters()
    if feature_extract:
        print('Feature extracting!')
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
    return model

## Helper Functions

def remap_classes(model, dataset):
    model.class_to_idx = dataset.class_to_idx
    model.idx_to_class = {
        idx: class_ for class_, idx in model.class_to_idx.items()
        }
    return model


def set_parameter_requires_grad(model):
    '''
    uncomment if you want to unfreeze deeper layers
    '''
    for param in model.parameters():
        param.requires_grad = False
        # for name, child in model.named_children():
        #     if name in ['layer4', 'avgpool', 'fc']:
        #         print(name + ' is unfrozen')
        #         for param in child.parameters():
        #             param.requires_grad = True
        #     else:
        #         print(name + ' is frozen')
        #         for param in child.parameters():
        #             param.requires_grad = False



def class_accuracy(model, valid_idx, valid_loader, topk=(1, 5)):
    """Measure the average accuracy per class

    Params
    --------
        model: trained model
        idx: length of validation loader, since using we are using subset, we have to use split idx
        valid_loader: validation dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    # Hold accuracy results
    acc_results = np.zeros((len(valid_idx), len(topk)))
    i = 0

    with torch.no_grad():
        for data, targets in valid_loader:
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')

            out = model(data)

            # Iterate through each example
            for pred, true in zip(out, targets):
                acc_results[i, :] = accuracy(
                    pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                i += 1

    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    classes = [x[11:] for x in classes]
    results['class'] = classes
    results = results.groupby(classes).mean()
    results = results.reset_index().rename(columns={'index': 'class'})
    results.to_csv('class_accuracy.csv', index=False)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(model, path):
    """
    Save a PyTorch model checkpoint.
    Model: model to save
    path: location to save model

    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50', 'resnet50-places'
                           ]), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

    # Extract the final classifier and the state dictionary
    if model_name == 'vgg16':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'resnet50' or model_name =='resnet50-places':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

    # attach the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)


def load_checkpoint(path):
    """
    Load a PyTorch model checkpoint
    """

    model_name = path.split('-')[0]
    assert (model_name in ['vgg16', 'resnet50'
                           ]), "Path must have the correct model name"

    checkpoint = torch.load(path)

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'resnet50' or model == 'resnet50-places':
        model = models.resnet50(pretrained=True)
        # Make sure to set parameters as not trainable
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    if train_on_gpu:
        model = model.to('cuda')

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer


def random_erase(p, area_ratio_range, min_aspect_ratio, max_attempt=20):
    '''
    Data Augmention Random Transformer based on Zhong, Zhun, et al. "Random Erasing Data Augmentation." arXiv preprint arXiv:1708.04896 (2017)
    '''

    sl, sh = area_ratio_range
    rl, rh = min_aspect_ratio, 1. / min_aspect_ratio

    def _random_erase(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]
        image_area = h * w

        for _ in range(max_attempt):
            mask_area = np.random.uniform(sl, sh) * image_area
            aspect_ratio = np.random.uniform(rl, rh)
            mask_h = int(np.sqrt(mask_area * aspect_ratio))
            mask_w = int(np.sqrt(mask_area / aspect_ratio))

            if mask_w < w and mask_h < h:
                x0 = np.random.randint(0, w - mask_w)
                y0 = np.random.randint(0, h - mask_h)
                x1 = x0 + mask_w
                y1 = y0 + mask_h
                image[y0:y1, x0:x1] = np.random.uniform(0, 1)
                break

        return image

    return _random_erase


def make_predictions(model, dataloader):
    '''
    make predictions for confusion matrix
    '''

    y_pred = []
    y_true = []
    with torch.no_grad():

        # Testing loop
        for data, targets in dataloader:

            # Tensors to gpu
            if train_on_gpu:
                data, targets = data.to('cuda'), targets.to('cuda')

            # Raw model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                output = pred.unsqueeze(0).to('cuda')

                pred = output.topk(1, dim=1, largest=True, sorted=True)[1]
                y_pred.append(pred.item())
                y_true.append(true.item())
        predictions = pd.DataFrame(
            {'y_pred': y_pred,
             'y_true': y_true
             })
        predictions.to_csv('predictions.csv', index=False)



def make_weights_for_balanced_classes(images, nclasses):
    '''
    function to weight balanced training classes
    '''

    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                
    return torch.DoubleTensor(weight) 



if __name__ == '__main__':
    main()



