from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os, torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, split_txt, transform=None):
        """
        root_dir: dataset path
        split_txt: dataset split, train or test.
        """
        self.image_list = []
        self.id_list = []
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = 0

        with open(split_txt, 'r') as f:
            line = f.readline()

            while line:
                img_name = line.split()[0]
                label = int(line.split()[1])

                self.image_list.append(img_name)
                self.id_list.append(label)
                line = f.readline()
        self.num_classes = max(self.id_list)+1
        # print(self.num_classes)
        
    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        label = self.id_list[idx]
        img_name = os.path.join(self.root_dir, img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)
        return image, label


def CIFAR100(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/CIFAR100/"
    train_txt = "./Dataset/VTAB-1K/Natural/CIFAR100/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/CIFAR100/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 100

    return train_data, test_data, num_class


def Caltech101(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/Caltech101/"
    train_txt = "./Dataset/VTAB-1K/Natural/Caltech101/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/Caltech101/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 102

    return train_data, test_data, num_class


def DTD(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/DTD/"
    train_txt = "./Dataset/VTAB-1K/Natural/DTD/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/DTD/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 47

    return train_data, test_data, num_class


def oxford_flowers102(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/oxford_flowers102/"
    train_txt = "./Dataset/VTAB-1K/Natural/oxford_flowers102/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/oxford_flowers102/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 102

    return train_data, test_data, num_class


def oxford_iiit_pet(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/oxford_iiit_pet/"
    train_txt = "./Dataset/VTAB-1K/Natural/oxford_iiit_pet/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/oxford_iiit_pet/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 37

    return train_data, test_data, num_class


def SVHN(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/SVHN/"
    train_txt = "./Dataset/VTAB-1K/Natural/SVHN/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/SVHN/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 10

    return train_data, test_data, num_class


def SUN397(target_size=224):
    path = "./Dataset/VTAB-1K/Natural/SUN397/"
    train_txt = "./Dataset/VTAB-1K/Natural/SUN397/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Natural/SUN397/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 397

    return train_data, test_data, num_class


# Specialized
def Patch_Camelyon(target_size=224):
    path = "./Dataset/VTAB-1K/Specialized/Patch_Camelyon/"
    train_txt = "./Dataset/VTAB-1K/Specialized/Patch_Camelyon/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Specialized/Patch_Camelyon/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 2

    return train_data, test_data, num_class


def Eurosat(target_size=224):
    path = "./Dataset/VTAB-1K/Specialized/Eurosat/"
    train_txt = "./Dataset/VTAB-1K/Specialized/Eurosat/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Specialized/Eurosat/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 10

    return train_data, test_data, num_class


def Resisc45(target_size=224):
    path = "./Dataset/VTAB-1K/Specialized/Resisc45/"
    train_txt = "./Dataset/VTAB-1K/Specialized/Resisc45/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Specialized/Resisc45/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 45

    return train_data, test_data, num_class


def Retinopathy(target_size=224):
    path = "./Dataset/VTAB-1K/Specialized/Retinopathy/"
    train_txt = "./Dataset/VTAB-1K/Specialized/Retinopathy/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Specialized/Retinopathy/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 5

    return train_data, test_data, num_class


# Structured
def Clevr_Count(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Clevr_Count/"
    train_txt = "./Dataset/VTAB-1K/Structured/Clevr_Count/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Clevr_Count/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 8

    return train_data, test_data, num_class


def Clevr_Dist(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Clevr_Dist/"
    train_txt = "./Dataset/VTAB-1K/Structured/Clevr_Dist/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Clevr_Dist/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 6

    return train_data, test_data, num_class


def DMLab(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/DMLab/"
    train_txt = "./Dataset/VTAB-1K/Structured/DMLab/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/DMLab/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 6

    return train_data, test_data, num_class


def Kitti(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Kitti/"
    train_txt = "./Dataset/VTAB-1K/Structured/Kitti/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Kitti/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 4

    return train_data, test_data, num_class


def Dsprites_loc(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Dsprites_loc/"
    train_txt = "./Dataset/VTAB-1K/Structured/Dsprites_loc/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Dsprites_loc/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 16

    return train_data, test_data, num_class


def Dsprites_ori(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Dsprites_ori/"
    train_txt = "./Dataset/VTAB-1K/Structured/Dsprites_ori/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Dsprites_ori/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 16

    return train_data, test_data, num_class


def Smallnorb_azi(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Smallnorb_azi/"
    train_txt = "./Dataset/VTAB-1K/Structured/Smallnorb_azi/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Smallnorb_azi/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 18

    return train_data, test_data, num_class


def Smallnorb_ele(target_size=224):
    path = "./Dataset/VTAB-1K/Structured/Smallnorb_ele/"
    train_txt = "./Dataset/VTAB-1K/Structured/Smallnorb_ele/train800val200.txt"
    test_txt = "./Dataset/VTAB-1K/Structured/Smallnorb_ele/test.txt"

    train_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = T.Compose([
        T.Resize(target_size, interpolation=3),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = CustomDataset(split_txt=train_txt, root_dir=path, transform=train_transforms)
    test_data = CustomDataset(split_txt=test_txt, root_dir=path, transform=test_transforms)
    num_class = 9

    return train_data, test_data, num_class


# train_data, test_data, num_class = Smallnorb_ele()
# print(len(train_data), len(test_data), num_class)

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 8, 8
# for i in range(1, cols * rows +1):
#     sample_idx = torch.randint(len(test_data), size=(1, )).item()
#     img, label = test_data[sample_idx]
#     plt.subplot(cols, rows, i)
#     plt.imshow(img.permute(1,2,0))
#     plt.title(str(label))
#     plt.axis('off')
# plt.savefig("./CIFAR100-2.png", dpi=800, bbox_inches='tight')