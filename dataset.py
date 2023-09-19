import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from variables import Config 
from torchvision import transforms

class CustomImageDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_filenames = os.listdir(images_dir)
        self.label_data = os.listdir(labels_dir)
        # print(f"images_dir: {self.images_dir}")
        # exit()

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # image_path = self.image_filenames[idx]
        # image = read_image(image_path)
        # label = self.label_data[idx]

        img_name = os.path.join(self.images_dir, self.image_filenames[idx])
        label_name = os.path.join(self.labels_dir, self.image_filenames[idx].replace('.jpg', '.txt'))

        image = read_image(img_name)
        
        # print(f"image_path: {img_name}")
        # exit()

        if os.path.exists(label_name):
             with open(label_name, 'r') as label_file:
                 label = label_file.read().strip()
                 label = int(label)  
        else:
             label = -1
        
        # print(f"image_path: {img_name}")
        # exit()

        if self.transform:
            image = self.transform(image)
        # print(type(image))

        return image, label

if __name__ == '__main__':
    pass