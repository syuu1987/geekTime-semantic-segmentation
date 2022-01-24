import torch
import numpy as np

from PIL import Image

img_size = (256, 256)
unet = torch.load('./weights/unet_epoch_51.pth')

unet.eval()


im = np.asarray(Image.open('data/JPEGImages/6.jpg').resize(img_size))

im = im / 255.
im = im.transpose(2, 0, 1)
im = im[np.newaxis, :, :]
im = im.astype('float32')
output = unet(torch.from_numpy(im)).detach().numpy()

output = np.squeeze(output)
output = np.where(output>0.5, 150, 0).astype(np.uint8)
print(output.shape, type(output))
im = Image.fromarray(output)
im.save('output.jpg')
