import csv
import os
from random import sample
import torch

from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

from .randaugment import RandAugmentMC
from pytorch_utils.dataloader import CsvDataset, get_denormalize_tensor
from matplotlib import pyplot as plt


CEBI_MEAN = (0.4913821518421173, 0.48098158836364746, 0.48094460368156433)
CEBI_STD = (0.1928231418132782, 0.2043113112449646, 0.2019609957933426)

train_labeled_csv_path = "/home/joe/data/datasets/cebi/Multiview_Sets/iteration3bn_fold0_train.csv"
# train_labeled_csv_path = "/home/joe/data/datasets/cebi/Multiview_Sets/HUGE_TEST.csv"
val_csv_path = "/home/joe/data/datasets/cebi/Multiview_Sets/iteration3bn_fold0_val.csv"
train_unlabeled_csv_path = "/home/joe/data/datasets/cebi/unlabeled/init.csv"

def get_cebi():
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CEBI_MEAN, std=CEBI_STD)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CEBI_MEAN, std=CEBI_STD)
    ])
   

    train_labeled_dataset = CsvDataset(
        train_labeled_csv_path,
        transform_labeled,
    )
    val_dataset = CsvDataset(
        val_csv_path,
        transform_val,
    )

    train_unlabeled_dataset = UnlabeledCsvDataset(
        train_unlabeled_csv_path,
        transform=TransformFixMatch(mean=CEBI_MEAN, std=CEBI_STD)
    )

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            # transforms.RandomCrop(size=224,
            #                   padding=int(224*0.125),
            #                   padding_mode='constant')
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((224,224)),
            # transforms.RandomCrop(size=224,
            #                   padding=int(224*0.125),
            #                   padding_mode='constant'),
            RandAugmentMC(n=2, m=10)
            ])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


# transform_weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=224,
#                               padding=int(224*0.125),
#                               padding_mode='constant'),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=cebi_mean, std=cebi_std)
# ],)

# transform_strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.RandomCrop(size=224,
#                               padding=int(224*0.125),
#                               padding_mode='constant'),
#             RandAugmentMC(n=2, m=10),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=cebi_mean, std=cebi_std)            
# ],)

class UnlabeledCsvDataset(Dataset):
    """
    Custom dataset based on csv files indicating samples and file-paths.
    File paths are supposed to be relative to the csv file location
    Compatible for single and multiview classification.
    Format:
        sample_id,view0_path,view1_path,view2_path,...
    
    """

    def __init__(self,csv_path , transform):
        """
        Arguments:
            csv_path : string
                Path to the csv file
            transform : torchvision.transform
                Agmentation transforms to be applied on each image
        """
        with open(csv_path,'r') as csv_file:
            reader = csv.reader(csv_file)
            _ = next(reader) # ditch header
            
            # gather samples
            samples = []
            sample_views = {}
            root = os.path.dirname(csv_path) # csv file location
            for row in reader:
                samples.append(row[0]) #sample_id
                # add all view paths
                view_paths = []
                for path in row[1:]:
                    # indicated path is absolute
                    if os.path.isabs(path):
                        view_paths.append(path)
                    # path is relative to csv location, generate absolute
                    else:
                        view_paths.append(
                            os.path.join(root,path)
                        )
                sample_views[row[0]] = view_paths

          
            self.samples = samples
            self.sample_views = sample_views
            self.loader = default_loader
            self.transform = transform

        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, path) where target is class_index of the target class.
        """
        key = self.samples[index]
        views = self.sample_views[key]
       
        sample_list = [self.transform(self.loader(path)) for path in views]

        weaks = torch.stack([x[0] for x in sample_list])
        strongs = torch.stack([x[1] for x in sample_list])
        # print(weaks.shape)
        # weaks = torch.stack
        # weak = torch.stack([transform_weak(self.loader(path))
        #                     for path in views])
        # strong = torch.stack([transform_weak(self.loader(path))
        #                     for path in views])

        # sample = (weaks_stacked,strongs_stacked)
        # create path that point to the sample 
        path = "/".join(views[0].split("/")[:-1]) + "/"  + str(key[1]) 

        
        # # Singleview
        # else:
        #     path = views[0]
        #     sample = self.transform(self.loader(path))


        return (weaks, strongs), path


def show_unalbeled_examples(dataloader, examples, mean, std):
    """
    Displays a grid of sample images from the dataset after augmentation

    Arguments
    ----------
    dataloader : torch.utils.data.DataLoader
        dataloader to take the samples from
    example_rows : int
        number of rows displayed
    example_cols : int
        number of collums displayed
    mean : list of floats
        mean used for normalizing the dataset
    std : list of floats
        standard deviation used for normalizing the dataset

    """
    denorm = get_denormalize_tensor(mean, std, False)
    (weaks,strongs), paths = next(iter(dataloader))
    example_cols = weaks[0].shape[0]*2
    example_rows = examples
    # Show a grid of example images
    fig, axes = plt.subplots(
        example_rows, example_cols, figsize=(
            19, 15))
    axes = axes.flatten()

    i = 0
    for weak,strong, path in zip(weaks,strongs, paths):
        images = torch.cat((weak,strong),0)
        for image in images:
            # denormalize the image
            image = denorm(image)
            image = image.numpy()
            ax = axes[i]

            if image.shape[0] == 1:
                plt.gray()
                image = image.squeeze()
            else:
                image = image.transpose(1, 2, 0)
            ax.imshow(image)
            ax.set_axis_off()
            ax.set_title(path, fontsize=7, color=(0, 0.2, 0))
            i += 1
        # stop loop
        if i >= example_rows * example_cols:
            break

    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.suptitle('Augmented training set images', fontsize=20)
    plt.show()