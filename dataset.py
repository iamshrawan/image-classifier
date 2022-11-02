from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import json

class ClassifierDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as js:
            self.instances = json.load(js)
        self.transforms = transforms.Compose([
                                transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                        ])
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        ex = self.instances[index]
        im = Image.open(ex['X']).convert('RGB')
        im_t = self.transforms(im)
        label = ex['y']
        return {'image': im_t, 'label': label}
