import torch
import torch.nn as nn

class Block(nn.Module):

    def __init__(self, in_channels, features):
        super(Block, self).__init__()

        self.features = features

        self.conv1 = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )
        self.conv2 = nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )

    def forward(self, input):
        x = self.conv1(input)
        x = nn.BatchNorm2d(num_features=self.features)(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(num_features=self.features)(x)
        x = nn.ReLU(inplace=True)(x)

        return x

class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)
        self.conv_encoder_2 = Block(features, features * 2)
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        self.bottleneck = Block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)

        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)

        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        bottleneck = self.bottleneck(conv_encoder_4_2)

        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)

        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)

        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)

        return torch.sigmoid(self.conv(conv_decoder_1_3))
