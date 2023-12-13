# ref: https://github.com/aladdinpersson

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from torchvision.utils import save_image

# load vgg pretained model
model = models.vgg19(pretrained=True).features
device = torch.device("cuda" if torch.cuda.is_available else "cpu") 
image_size = 256

# VGG class
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.selected_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]
    
    def forward(self, x):
        features = []
        for n_layer, layer in enumerate(self.model):
            x = layer(x)
            if str(n_layer) in self.selected_features:
                features.append(x)
        return features 

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

loader = transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
    ]
)

# image configs
target_image = load_image("T2.jpg")
style_image = load_image("woman.jpg")
generated_image = target_image.clone().requires_grad_(True)

# network hyperparams
model = VGG().to(device).eval() # freeze weights
n_steps = 6000
lr = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated_image])

# training loop
for step in range(n_steps):
    generated_image_features = model(generated_image)
    target_image_features = model(target_image)
    style_image_feactures= model(style_image)

    style_loss = target_loss = 0

    for gen_feature, tar_feature, style_feature in zip(generated_image_features, target_image_features, style_image_feactures):
        batch_size, channel, height, width = gen_feature.shape
        target_loss += torch.mean((gen_feature - tar_feature)**2)

        # compute gram matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height * width).t()
        )

        A = style_feature.view(channel, height * width).mm(
            style_feature.view(channel, height * width).t()
        )

        style_loss += torch.mean((G-A)**2)
    
    total_loss = alpha*target_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


    count = 0
    if step % 200 == 0:
        print(total_loss)
        save_image(generated_image, "generated_image_{0}.png".format(count))
        count += 1