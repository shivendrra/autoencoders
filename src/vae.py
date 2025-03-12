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

class DownBlock(nn.Module):
  def __init__(self, channels_in, channels_out):
    super(DownBlock, self).__init__()
    self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)
    self.bn1 = nn.BatchNorm2d(channels_out)
    self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1)
    self.bn2 = nn.BatchNorm2d(channels_out)
    self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 2, 1)

  def forward(self, x):
    x_skip = self.conv3(x)
    x = F.elu(self.bn1(self.conv1(x)))
    x = self.conv2(x) + x_skip
    return F.elu(self.bn2(x))

class UpBlock(nn.Module):
  def __init__(self, channels_in, channels_out):
    super(UpBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(channels_in)
    self.conv1 = nn.Conv2d(channels_in, channels_in, 3, 1, 1)
    self.bn2 = nn.BatchNorm2d(channels_in)
    self.conv2 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
    self.conv3 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
    self.up_nn = nn.Upsample(scale_factor=2, mode="nearest")

  def forward(self, x_in):
    x = F.elu(self.bn2(x_in))
    x_skip = self.up_nn(self.conv3(x))
    x = self.up_nn(F.elu(self.bn2(self.conv1(x))))
    return self.conv2(x) + x_skip

class Encoder(nn.Module):
  def __init__(self, channels, ch=32, z=32):
    super(Encoder, self).__init__()
    self.conv_1 = nn.Conv2d(channels, ch, 3, 1, 1)
    self.conv_block1 = DownBlock(ch, ch)
    self.conv_block2 = DownBlock(ch, ch * 2)
    self.conv_block3 = DownBlock(ch * 2, ch * 4)

    # Instead of flattening (and then having to unflatten) out our feature map and 
    # putting it through a linear layer we can just use a conv layer
    # where the kernal is the same size as the feature map 
    # (in practice it's the same thing)
    self.conv_mu = nn.Conv2d(4 * ch, z, 4, 1)
    self.conv_logvar = nn.Conv2d(4 * ch, z, 4, 1)

  # this function will sample from our distribution
  def sample(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
    
  def forward(self, x):
    x = F.elu(self.conv_1(x))
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = self.conv_block3(x)

    mu = self.conv_mu(x)
    logvar = self.conv_logvar(x)
    x = self.sample(mu, logvar)

    return x, mu, logvar

class Decoder(nn.Module):
  def __init__(self, channels, ch = 32, z = 32):
    super(Decoder, self).__init__()
    self.conv1 = nn.ConvTranspose2d(z, 4 * ch, 4, 1)
    self.conv_block1 = UpBlock(4 * ch, 2 * ch)
    self.conv_block2 = UpBlock(2 * ch, ch)
    self.conv_block3 = UpBlock(ch, ch)
    self.conv_out = nn.Conv2d(ch, channels, 3, 1, 1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv_block1(x)
    x = self.conv_block2(x)
    x = F.elu(self.conv_block3(x))

    return torch.tanh(self.conv_out(x))

class VAE(nn.Module):
  def __init__(self, channel_in, ch=16, z=32):
    super(VAE, self).__init__()
    self.encoder = Encoder(channels=channel_in, ch=ch, z=z)
    self.decoder = Decoder(channels=channel_in, ch=ch, z=z)

  def forward(self, x):
    encoding, mu, logvar = self.encoder(x)  
    # only sample during training or when we want to generate new images
    # just use mu otherwise
    if self.training:
        x = self.decoder(encoding)
    else:
        x = self.decoder(mu)
    return x, mu, logvar

batch_size = 64
lr = 1e-4
nepoch = 10
noise_scale = 0.3
latent_size = 128
root = '../dataset'

vae_net = VAE(channel_in=1, z=latent_size).to("cpu")
optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))

transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])])

train_set = Datasets.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=4)

test_set = Datasets.MNIST(root=root, train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

loss_log = []
train_loss = 0

def vae_loss(recon, x, mu, logvar):
  recon_loss = F.mse_loss(recon, x)  
  # Here is our KL divergance loss implemented in code
  # We will use the mean across the dimensions instead of the sum (which is common and would require different scaling)
  kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
  # We'll tune the "strength" of KL divergance loss to get a good result 
  loss = recon_loss + 0.1 * kl_loss
  return loss

pbar = trange(0, nepoch, leave=False, desc="Epoch")   
vae_net.train()
train_loss = 0
for epoch in pbar:
  pbar.set_postfix_str('Loss: %.4f' % (train_loss/len(train_loader)))
  train_loss = 0
  for i, data in enumerate(tqdm(train_loader, leave=False, desc="Training")):
    image = data[0].to("cpu")
    # Forward pass the image in the data tuple
    recon_data, mu, logvar = vae_net(image)
    
    # Calculate the loss
    loss = vae_loss(recon_data, image, mu, logvar)
    
    # Log the loss
    loss_log.append(loss.item())
    train_loss += loss.item()
    # Take a training step
    vae_net.zero_grad()
    loss.backward()
    optimizer.step()