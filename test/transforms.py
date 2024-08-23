from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2, PILToTensor
from torchvision.datasets import EMNIST

plt.rcParams["savefig.bbox"] = 'tight'

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(2)

# If you're trying to run that on collab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
#from helpers import plot

from kornia.morphology import erosion, dilation
def thicken(image:torch.Tensor) -> torch.Tensor:
    image = torch.unsqueeze(image, 0)
    t = torch.randint(1, 3, (2,))
    kernel = torch.ones((t[0], t[1]))
    if np.random.rand() < 0.5:
        return torch.squeeze(erosion(image, kernel=kernel), 0)
    else:
        return torch.squeeze(dilation(image, kernel=kernel), 0)
def fill_nan(image:torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(image, nan=-0.42421296)

transformer = transforms.Compose([
                               transforms.ToTensor(),
                               v2.Lambda(thicken),
        ])
augment = torch.nn.Sequential(
            v2.ElasticTransform(alpha=30.0, sigma=3.0),
            v2.RandomPerspective(),
            v2.RandomAffine(30, translate=(0.1, 0.1)),
            v2.Normalize((0.1307,), (0.3081,)),
            v2.Lambda(fill_nan)
            )
dataset = EMNIST(root='.venv/data',
                split='bymerge',
                train=False,
                download=True,
                transform=transformer)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
iterator = iter(dataloader)
#transformed_imgs = [np.asarray(torch.squeeze(transformer(orig_img), 0)) for _ in range(31)]


#print(orig_img.size())

img, target = next(iterator)
img2 = augment(img)
print(img.size())
    
fig = plt.figure()
for i in range(img.size()[0]):
    fig.add_subplot(4, 8, i*2+1) 
    plt.imshow(np.asarray(img[i, 0, :, :]))
    fig.add_subplot(4, 8, i*2+2) 
    plt.imshow(np.asarray(img2[i, 0, :, :]))
plt.show()