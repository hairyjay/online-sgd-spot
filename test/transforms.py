from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
from torchvision.transforms import v2, PILToTensor
from torchvision.datasets import MNIST

plt.rcParams["savefig.bbox"] = 'tight'

# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(0)

# If you're trying to run that on collab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
#from helpers import plot

dataset = MNIST(root='.venv/data',
                train=False,
                download=True)
orig_img, label = dataset[7]
totensor = PILToTensor()

from kornia.morphology import erosion, dilation
def thicken(image:torch.Tensor) -> torch.Tensor:
    image = torch.unsqueeze(image, 0)
    t = torch.randint(1, 3, (2,))
    kernel = torch.ones((t[0], t[1]))
    if np.random.rand() < 0.5:
        return torch.squeeze(erosion(image, kernel=kernel), 0)
    else:
        return torch.squeeze(dilation(image, kernel=kernel), 0)
    

#transformer = v2.ElasticTransform(alpha=30.0, sigma=3.0)
transformer = v2.Lambda(thicken)
transformed_imgs = [np.asarray(torch.squeeze(transformer(totensor(orig_img)), 0)) for _ in range(31)]
#print(orig_img.size())

fig = plt.figure() 
fig.add_subplot(4, 8, 1) 
plt.imshow(np.asarray(orig_img))
for i, img in enumerate(transformed_imgs):
    fig.add_subplot(4, 8, i+2) 
    plt.imshow(np.asarray(img))
plt.show()