import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from vit_pytorch import ViT
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
from gener import Generator
import torch.optim as optim
import torchvision.utils as vutils
import matplotlib.animation as animation
from IPython.display import HTML

image_size = 32
batch_size = 8

discriminator = ViT(
    image_size = image_size,
    patch_size =  4,
    num_classes = 1,
    dim = 384,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
    )
    
# image_size = 64
# dataroot = "celeb/"

# dataset = dset.ImageFolder(root=dataroot,
                           # transform=transforms.Compose([
                               # transforms.Resize(image_size),
                               # transforms.CenterCrop(image_size),
                               # transforms.ToTensor(),
                               # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           # ]))
 
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         # shuffle=True)
                                         
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])                                         

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True)
# for i in dataloader:
    # print(i[0][1].shape)
    # plt.figure(figsize=(10,5))
    # plt.imshow(np.transpose(i[0][0],(1,2,0)))
    # plt.show()
    # break
    
# fj

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

nz = 100
fixed_noise = torch.randn(8,nz).to(device)
discriminator = discriminator.to(device)
gener = Generator().to(device)


criterion = nn.HingeEmbeddingLoss()

lr = 0.0002
beta1 = 0.5
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(gener.parameters(), lr=lr, betas=(beta1, 0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 4
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        dat = data[0].to(device)
        b_size = dat.size(0)
        
        discriminator.zero_grad()
        # for i in range(5):
        label = torch.full((b_size,), 1, dtype=torch.float, device=device)
        output = discriminator(dat).view(-1)
        # print(output.shape)
        # print(label.shape)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()
        
        noise = torch.randn(b_size, nz, device = device)
        fake = gener(noise)
        label.fill_(-1)
        output = discriminator(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        optimizerD.step()
        
        
        gener.zero_grad()
        label.fill_(1)
        output = discriminator(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        
        
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = gener(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
        
        
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


plt.figure(figsize=(10,5))
for i in img_list:
    plt.imshow(np.transpose(i,(1,2,0)))
    plt.show()
    
 



    

    


