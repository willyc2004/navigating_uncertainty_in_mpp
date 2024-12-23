from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, obs):
        hidden, init_embed = self.encoder(obs)
        dec_out = self.decoder(obs, hidden)
        return dec_out