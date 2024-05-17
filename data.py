import os
import torch
import argparse
import skimage.io
from torchvision import transforms

def load_files_from_dir(root_dir):
    '''Iterate over directories to get images' paths'''
    file_paths = []

    for dir_name, _, file_names in os.walk(root_dir):
   
        for file_name in file_names:
    
            if file_name != ".DS_Store":

                file_path = os.path.join(dir_name, file_name)

                file_paths.append(file_path)
    return file_paths

class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform=False):
        """
        Initializes a new Dataset
        Args:
            paths (list/array):  of images' paths
            transform (torchvision.transforms): preprocessing
        """
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, x):
        
        x = self.paths[x]
        img = skimage.io.imread(x)
        if img.shape[-1] == 4:
            # Remove the alpha channel
            img = img[:, :, :3]
            
        image = self.transform(img)
        
        '''extracting second from end entity of spitted path, last is img number, 
           second last is directory (one directory indicates one person)'''
        person_id = x.split('/')[-2]
        
        return image, person_id

def Norm(img):
    return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224), antialias=True), 
                                transforms.ConvertImageDtype(torch.float32), 
                                transforms.Lambda(Norm)])

if __name__ == "__main__":
    '''Used by me dataset is: https://microsoft.github.io/DigiFace1M/'''
    
    parser = argparse.ArgumentParser(description='Creat Datasets/Dataloader')
    parser.add_argument('--paths', type=str, default='subjects_100000-133332_5_imgs'
                        , help="source path to your dataset")
    parser.add_argument('--target_path', type=str, default='faces_dataloader.pt', help="destination path for dataloader")
    
    args = parser.parse_args()

    paths = load_files_from_dir(args.paths)
    dataset = FaceDataset(paths, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=5) # is number of samples per person
    torch.save(dataloader, args.target_path)