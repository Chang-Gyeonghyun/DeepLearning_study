import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.dataloader import get_data_loaders
from models.generator import Generator
from models.discriminator import Discriminator
from utils.weights_init import weights_init
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = Generator().to(DEVICE)
    netG.apply(weights_init)

    netD = Discriminator().to(DEVICE)
    netD.apply(weights_init)

    train_loader = get_data_loaders()

    criterion = nn.BCELoss()

    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    epochs = 5

    print("Starting Training Loop...")
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):

            # (1) Update the discriminator with real data
            netD.zero_grad()
            real_cpu = data[0].to(DEVICE)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), 1., dtype=torch.float, device=DEVICE)
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # (2) Update the discriminator with fake data
            noise = torch.randn(b_size, 100, 1, 1, device=DEVICE)
            fake = netG(noise)
            label.fill_(0.)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (3) Update the generator with fake data
            netG.zero_grad()
            label.fill_(1.)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, epochs, i, len(train_loader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    torch.save(netG.state_dict(), './saved/generator.pth')
