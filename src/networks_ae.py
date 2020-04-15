import torch.nn as nn
import torch.nn.functional as F


def select_network(network,decode_dim=10):
    if network == 'linear':
        net = autoencoder_linear(decode_dim)
    elif network == 'conv':
        net = autoencoder()
    else:
        raise ValueError
    return net


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),  # b, 16, 28, 28
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=3, padding=1),  # b, 32, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 32, 5, 5
            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # b, 32, 5, 5
            nn.ReLU(True),
            nn.Conv2d(32, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return enc,dec

class autoencoder_linear(nn.Module):
    def __init__(self,decode_dim=10):
        super(autoencoder_linear, self).__init__()
        self.decode_dim = decode_dim
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.Tanh(),
            nn.Linear(256, self.decode_dim),
            nn.Tanh())
        self.decoder = nn.Sequential(
            nn.Linear(self.decode_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 28 * 28),
            nn.Tanh())

    def forward(self, x):
        x = x.view(-1,28*28)
        enc = self.encoder(x)
        dec = self.decoder(enc)
        dec = dec.reshape(-1,1,28,28)
        return enc,dec


# class autoencoder_linear(nn.Module):
#     def __init__(self):
#         super(autoencoder_linear, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 256),
#             nn.Tanh(),
#             nn.Linear(256, 2),
#             nn.Tanh())
#         self.decoder = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.Tanh(),
#             nn.Linear(256, 28 * 28),
#             nn.Tanh())
#
#     def forward(self, x):
#         x = x.view(-1,28*28)
#         enc = self.encoder(x)
#         dec = self.decoder(enc)
#         dec = dec.reshape(-1,1,28,28)
#         return enc,dec
#

class AutoEncoder_v3(nn.Module):

    def __init__(self, code_size: object) -> object:
        self.imgsize = 784
        self.width = 28
        self.height = 28
        super().__init__()
        self.code_size = code_size

        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)

        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, self.imgsize)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return code, out

    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))

        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))

        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, self.width, self.height])
        return out