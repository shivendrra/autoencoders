import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from tqdm import tqdm, trange

LEARNING_RATE = 0.0005
device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
  def __init__(self, channels, ch=32, z=32):
    super().__init__()
    self.conv1 = nn.Conv2d(channels, ch, 3, 2, 1)
    self.bn1 = nn.BatchNorm2d(ch)
    self.conv2 = nn.Conv2d(ch, 2 * ch, 3, 2, 1)
    self.bn2 = nn.BatchNorm2d(2 * ch)
    self.conv3 = nn.Conv2d(2 * ch, 4 * ch, 3, 2, 1)
    self.bn3 = nn.BatchNorm2d(4 * ch)
    self.conv_out = nn.Conv2d(4 * ch, z, 4, 1)
        
  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    return self.conv_out(x)

class Decoder(nn.Module):
  def __init__(self, channels, ch = 32, z = 32):
    super().__init__()
    self.conv1 = nn.ConvTranspose2d(z, 4 * ch, 4, 1)
    self.bn1 = nn.BatchNorm2d(4 * ch)
    self.conv2 = nn.ConvTranspose2d(4 * ch, 2 * ch, 3, 2, 1, 1)
    self.bn2 = nn.BatchNorm2d(2 * ch)
    self.conv3 = nn.ConvTranspose2d(2 * ch, ch, 3, 2, 1, 1)
    self.bn3 = nn.BatchNorm2d(ch)
    self.conv4 = nn.ConvTranspose2d(ch, ch, 3, 2, 1, 1)
    self.bn4 = nn.BatchNorm2d(ch)
    self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))
    x = F.relu(self.bn4(self.conv4(x)))
    return torch.tanh(self.conv_out(x))

class AE(nn.Module):
  def __init__(self, channel_in, ch=16, z=32):
    super(AE, self).__init__()
    self.encoder = Encoder(channels=channel_in, ch=ch, z=z)
    self.decoder = Decoder(channels=channel_in, ch=ch, z=z)

  def forward(self, x):
    encoding = self.encoder(x)
    x = self.decoder(encoding)
    return x, encoding

batch_size = 64
lr = 1e-4
nepoch = 10
noise_scale = 0.3
latent_size = 128
root = '../dataset'

# defining the transforms, sampling 32x32 images, coz it's easier for model to reconstruct
transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

train_set = Datasets.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=4)

test_set = Datasets.MNIST(root=root, train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

# creating our network
ae_net = AE(channel_in=1, z=latent_size).to(device)
optimizer = optim.Adam(ae_net.parameters(), lr=lr)
loss_func = nn.MSELoss()

dataiter = iter(test_loader)
test_images = dataiter.next()[0]
recon_data, encoding = ae_net(test_images.to(device))

# noisy img saml=pling
random_sample = (torch.bernoulli((1 - noise_scale) * torch.ones_like(test_images)) * 2) - 1
noisy_test_img = random_sample * test_images

loss_log, train_loss = [], 0
pbar = trange(0, nepoch, leave=False, desc="Epoch")    
for epoch in pbar:
  pbar.set_postfix_str('Loss: %.4f' % (train_loss/len(train_loader)))
  train_loss = 0
  for i, data in enumerate(tqdm(train_loader, leave=False, desc="Training")):
    image = data[0].to(device)
    random_sample = (torch.bernoulli((1 - noise_scale) * torch.ones_like(image)) * 2) - 1
    noisy_img = random_sample * image

    # forward pass the image in the data tuple
    recon_data, _ = ae_net(noisy_img)
    # calculate the MSE loss
    loss = loss_func(recon_data, image)    
    # loggin the loss
    loss_log.append(loss.item())
    train_loss += loss.item()

    # taking a training step
    ae_net.zero_grad()
    loss.backward()
    optimizer.step()

# ground turth for visualization, not implemented though!
out = vutils.make_grid(test_images[0:8], normalize=True)
# noisy truth for visualization
out = vutils.make_grid(noisy_test_img[0:8], normalize=True)

# reconstruction (encoder)
recon_data, encoding = ae_net(noisy_test_img.to(device))
out = vutils.make_grid(recon_data.detach().cpu()[0:8], normalize=True) # not implemented

# reconstruction (decoder)
recon_data = ae_net.decoder(encoding.std(0, keepdims=True) * torch.randn_like(encoding) + encoding.mean(0, keepdims=True))
out = vutils.make_grid(recon_data.detach().cpu()[0:8], normalize=True) # no implemented