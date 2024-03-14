import torch
import torchvision.utils as vutils
from models.generator import Generator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Generator().to(device)
    model.load_state_dict(torch.load('./saved/generator.pth', map_location=device))

    with torch.no_grad():
        fake = model(torch.randn(64, 100, 1, 1, device=device)).detach().cpu()

    fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
    plt.imshow(fake_grid.permute(1, 2, 0))
    plt.axis('off')
    plt.show()
