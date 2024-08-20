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
#torch.manual_seed(0)

from kornia.morphology import erosion, dilation
def thicken(image:torch.Tensor) -> torch.Tensor:
    image = torch.unsqueeze(image, 0)
    t = torch.randint(1, 3, (2,))
    kernel = torch.ones((t[0], t[1]))
    if np.random.rand() < 0.5:
        return torch.squeeze(erosion(image, kernel=kernel), 0)
    else:
        return torch.squeeze(dilation(image, kernel=kernel), 0)

transform = transforms.Compose([
                                transforms.ToTensor(),
                                v2.Lambda(thicken),
                                v2.ElasticTransform(alpha=30.0, sigma=3.0),
                                transforms.RandomPerspective(),
                                transforms.RandomAffine(30, translate=(0.1, 0.1)),
                                transforms.Normalize((0.1307,), (0.3081,))
            ])
dataset = EMNIST(root='.venv/data',
                split='bymerge',
                train=True,
                download=True,
                transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
iterator = iter(dataloader)
i = 0

while True:
    i += 1
    if i % 10000 == 0:
        print("{}/{}".format(i*1, len(dataset)))
    try:
        img, target = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        img, target = next(iterator)
    #if torch.isnan(img).any():
        #print("NaN FOUND: i = {}, seed = {}".format(i, torch.seed()))
    img = torch.squeeze(img, 0)
    print(img.numpy())
    #print(label)
    #img = torch.squeeze(img, 0)
    plt.imshow(img.numpy())
    plt.show()
    break