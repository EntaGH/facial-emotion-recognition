import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

emotional_classes = [ 'happy', 'sad', 'surprise', 'neutral']

transform = transforms.Compose([
    transforms.Resize((48, 48)),       # Resize image to 48x48
    transforms.ToTensor(),               # Convert to tensor [0, 1]
    # Optional: Normalize (e.g., for pretrained models like ResNet)
])

#load dataset
def get_train_data(file_path,  batch_size = 64, samples_per_class = 3000):
    images = []
    for idx, item in enumerate(emotional_classes):
        class_images = []
        for image in os.listdir(os.path.join(file_path, 'train', item)):
            path = os.path.join(os.path.join(file_path,'train',item,image))
            try:
                img = Image.open(path)
                img = transform(img)
                class_images.append({'array' : img.reshape((1,48,48)), 'label' : idx, 'text': item})
            except Exception as e:
                print('Issue with: {}'.format(path))
                print(f'Error: {e}')
        original_len = len(class_images)
        if original_len < samples_per_class:
            repeat_count = samples_per_class // original_len
            remainder = samples_per_class % original_len
            class_images = class_images * repeat_count + class_images[:remainder]
        else:
            class_images = class_images[:samples_per_class]  # Truncate if more

        images.extend(class_images)
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=True)
    return dataloader

def get_test_data(file_path,  batch_size = 64):
    images = []
    for idx, item in enumerate(emotional_classes):
        for image in os.listdir(os.path.join(file_path, 'test', item)):
            path = os.path.join(os.path.join(file_path,'test',item,image))
            try:
                img = Image.open(path)
                img = transform(img)
                images.append({'array' : img.reshape((1,48,48)), 'label' : idx, 'text': item})
            except Exception as e:
                print('Issue with: {}'.format(path))
                print(f'Error: {e}')
    dataloader = DataLoader(images, batch_size=batch_size, shuffle=True)
    return dataloader


# if __name__=="__main__":
#     dataloader = get_train_data('data')
#     print(len(dataloader))
    # for batch in dataloader:
    #     text = batch['label']
    #     array = batch["array"]
    #     print('Label: ',text[0])
    #     print("Audio batch shape:", array[0].shape)
    #     print("Input IDs batch shape:", array[0])
    #     # breakpoint() # Useful for debugging the batch content
    #     break

