import torch
import torch.nn as nn
import torch.optim as optim
from kornia.losses import ssim_loss
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import os
import warnings
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from requests.exceptions import RequestsDependencyWarning


# normalization

def adaptive_normalize(image):
    min_val = image.min()
    max_val = image.max()
    return (image - min_val) / (max_val - min_val + 1e-5)


# Attention block:

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


# Maxout

class Maxout(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c // self.pool_size, self.pool_size, h, w)
        return x.max(dim=2)[0]


# CS_DNET

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



# Dataset

class HazeDataset(Dataset):
    def __init__(self, hazy_dir, target_dir, patch_size=64, transform=None):
        self.hazy_dir = hazy_dir
        self.target_dir = target_dir
        self.transform = transform
        self.patch_size = patch_size
        self.files = sorted(os.listdir(hazy_dir))

    def __len__(self):
        return len(self.files)

    def extract_patches(self, image):
        patches = []
        w, h = image.size
        for i in range(0, w, self.patch_size):
            for j in range(0, h, self.patch_size):
                p = image.crop((i, j, i+self.patch_size, j+self.patch_size))
                if p.size == (self.patch_size, self.patch_size):
                    patches.append(p)
        return patches

    def __getitem__(self, idx):
        hazy = Image.open(os.path.join(self.hazy_dir, self.files[idx])).convert("RGB")
        target = Image.open(os.path.join(self.target_dir, self.files[idx])).convert("RGB")

        hazy_p = self.extract_patches(hazy)
        target_p = self.extract_patches(target)

        k = torch.randint(0, len(hazy_p), (1,)).item()

        if self.transform:
            hazy_p[k] = self.transform(hazy_p[k])
            target_p[k] = self.transform(target_p[k])

        return hazy_p[k], target_p[k]


# Loss

def gradient_loss(pred, target):
    dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    dx_gt = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    dy_gt = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    return torch.mean(torch.abs(dx - dx_gt)) + torch.mean(torch.abs(dy - dy_gt))

# Train

def train(model, train_loader, val_loader, optimizer, device, epochs):
    criterion = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    best = 1e9

    for ep in range(epochs):
        model.train()
        tr_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            #loss = 0.5 * criterion(out, y) + 0.3 * ssim_loss(out, y[:, 0:1, :, :], 5) + 0.2 * gradient_loss(out, y)
            loss = 0.9*criterion(out, y) + 0.1 * gradient_loss(out, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                #loss = 0.5 * criterion(out, y) + 0.3 * ssim_loss(out, y[:,0:1,:,:],5) + 0.2 * gradient_loss(out, y)
                loss = 0.9 * criterion(out, y) + 0.1 * gradient_loss(out, y)
                val_loss += loss.item()

        tr_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {ep+1:03d} | Train {tr_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), "RGB_NIR_best_model_outdoor.pth")
            print(f"Best Model is saved at {ep + 1}. epoch (Validation Loss: {val_loss:.4f})")

# Main Function

def main():
    warnings.filterwarnings("ignore")
    warnings.simplefilter("ignore", RequestsDependencyWarning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([


        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),


        transforms.ToTensor(),  # To tensor
        #transforms.Lambda(normalize_16bit)
        #transforms.Lambda(adaptive_normalize)
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization
    ])

    dataset = HazeDataset(
        hazy_dir=r"HazyDir",
        target_dir = r"TargetDir",

        patch_size=64,
        transform=transform
    )

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)

    model = CSDNET().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train(model, train_loader, val_loader, optimizer, device, epochs=300)

if __name__ == "__main__":
    main()
