import os
from torch.utils.data import Dataset, DataLoader
import cv2


class MyImageFolder(Dataset):
    def __init__(self, root_dir, both_transforms, highres_transform, lowres_transform):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)
        self.both_transforms = both_transforms
        self.highres_transform = highres_transform
        self.lowres_transform = lowres_transform

        for index, name in enumerate(self.class_names):
            files = os.listdir(os.path.join(root_dir, name))
            self.data += list(zip(files, [index] * len(files)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])

        image = cv2.imread(os.path.join(root_and_dir, img_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        both_transform = self.both_transforms(image=image)["image"]
        low_res = self.lowres_transform(image=both_transform)["image"]
        high_res = self.highres_transform(image=both_transform)["image"]
        return low_res, high_res


if __name__ == "__main__":
    dataset = MyImageFolder(root_dir="./data/")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print("Low Resolution image shape:  ", low_res.shape)
        print("High Resolution image shape: ", high_res.shape)
        break