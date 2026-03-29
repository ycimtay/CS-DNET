import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from piq import brisque

import numpy as np


from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips

# =========================================================
# SAME MODEL DEFINITIONS (MUST MATCH TRAIN)
# =========================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c // self.pool_size, self.pool_size, h, w)
        return x.max(dim=2)[0]



class CSDNET(nn.Module):
    def __init__(self):
        super().__init__()

        # Multi-scale
        self.conv3 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(3, 32, 5, padding=2)
        self.conv7 = nn.Conv2d(3, 32, 7, padding=3)
        self.conv9 = nn.Conv2d(3, 32, 9, padding=4)


        self.maxout1 = Maxout(pool_size=2)

        self.conv_refine1 = nn.Conv2d(64, 64, 3, padding=1)
        self.se1 = SEBlock(64)

        self.maxout2 = Maxout(pool_size=2)

        self.conv_refine2 = nn.Conv2d(32, 32, 3, padding=1)
        self.se2 = SEBlock(32)

        self.conv_out = nn.Conv2d(32, 1, 5, padding=2)

    def forward(self, x):
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        f9 = self.conv9(x)

        f = torch.cat([f3, f5, f7,f9], dim=1)
        f = self.maxout1(f)

        f = self.conv_refine1(f)
        f = self.se1(f)

        f = self.maxout2(f)

        f = self.conv_refine2(f)
        f = self.se2(f)

        out = self.conv_out(f)

        return out
def inference(
    model_path,
    image_path,
    device
):
    transform = transforms.ToTensor()
    import cv2
    import numpy as np


    model = CSDNET().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    img = Image.open(image_path).convert("RGB")
    img = img.resize((1920, 1080))
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)


    out = out.squeeze().cpu().numpy()
    out_np = np.clip(out, 0, 1)



    plt.figure()
    plt.imshow(img, cmap="gray")
    plt.title("RGB")
    plt.axis("off")

    plt.figure()
    plt.imshow(out_np, cmap="gray")
    plt.title("Output")
    plt.axis("off")

    plt.show()





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    inference(

        model_path=r"RGB_NIR_best_model_outdoor.pth",
        image_path=r"ImagePath",
        device=device
    )
