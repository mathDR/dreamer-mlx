import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # groups == channels, matching the original (per-channel / instance-style norm)
        self.norm = nn.GroupNorm(out_channels, out_channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C)
        x = self.conv(x)
        x = self.norm(x)
        return nn.silu(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        hid_channels: int = 64,
        num_spatial_layers: int = 4,
    ):
        super().__init__()
        self.blocks = [
            ConvBlock(in_channels if i == 0 else hid_channels, hid_channels)
            for i in range(num_spatial_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        # Input: (B, H, W, C)  ->  Output: (B, FlattenedDim)
        for block in self.blocks:
            x = block(x)
        return x.reshape(x.shape[0], -1)


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        hid_channels: int = 64,
        num_spatial_layers: int = 4,
        latent_size: int = 4,  # spatial size of the encoder's final feature map
    ):
        super().__init__()
        self.hid_channels = hid_channels
        self.latent_size = latent_size

        # Each layer exactly doubles H and W: (H-1)*2 - 2*1 + 3 + 1 = 2H
        self.blocks = [
            nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            )
            for _ in range(num_spatial_layers)
        ]
        self.readout = nn.Conv2d(hid_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # Input: (B, FlattenedDim)  ->  Output: (B, H, W, OutC)
        # NHWC flatten order, consistent with Encoder above
        x = x.reshape(x.shape[0], self.latent_size, self.latent_size, self.hid_channels)
        for block in self.blocks:
            x = nn.silu(block(x))
        return self.readout(x)


class VariationalEncoder(nn.Module):
    def __init__(self, in_channels=3, hid_channels=64, num_spatial_layers=4,
                 latent_size=4, latent_dim=128):
        super().__init__()
        self.trunk = Encoder(in_channels, hid_channels, num_spatial_layers)
        flat_dim = hid_channels * latent_size * latent_size
        self.mean_head = nn.Linear(flat_dim, latent_dim)
        self.std_head = nn.Linear(flat_dim, latent_dim)

    def __call__(self, x):
        h = self.trunk(x)                          # (B, flat_dim)
        mean = self.mean_head(h)                   # (B, latent_dim)
        std = nn.softplus(self.std_head(h)) + 1e-4 # strictly positive
        return mean, std


class VAEDecoder(nn.Module):
    def __init__(self, out_channels=3, hid_channels=64, num_spatial_layers=4,
                 latent_size=4, latent_dim=128):
        super().__init__()
        self.proj = nn.Linear(latent_dim, hid_channels * latent_size * latent_size)
        self.decoder = Decoder(out_channels, hid_channels, num_spatial_layers, latent_size)

    def __call__(self, z):
        return self.decoder(self.proj(z))


def sample(mean, std):
    # Reparameterization trick: gradients flow through mean and std
    return mean + std * mx.random.normal(mean.shape)


def kl_divergence(mean, std):
    kl = 0.5 * (mx.square(mean) + mx.square(std) - 1.0 - 2.0 * mx.log(std))
    return kl.sum(axis=-1).mean()


class VAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hid_channels=64,
                 num_spatial_layers=4, latent_size=4, latent_dim=128):
        super().__init__()
        self.encoder = VariationalEncoder(
            in_channels, hid_channels, num_spatial_layers, latent_size, latent_dim
        )
        self.decoder = VAEDecoder(
            out_channels, hid_channels, num_spatial_layers, latent_size, latent_dim
        )
 
    def __call__(self, x):
        mean, std = self.encoder(x)
        z = sample(mean, std)
        return self.decoder(z), mean, std

def loss_fn(model, x, beta):
    recon, mean, std = model(x)
    recon_loss = mx.square(recon - x).sum(axis=(1, 2, 3)).mean()
    kl = kl_divergence(mean, std)
    return recon_loss + beta * kl, recon_loss, kl   # flat tuple

def make_batch(batch_size, img_size, channels):
    coords = mx.linspace(0, 1, img_size)
    freq = mx.random.uniform(1, 4, (batch_size, 1, 1, channels))
    phase = mx.random.uniform(0, 6.28, (batch_size, 1, 1, channels))
    xx = coords.reshape(1, 1, img_size, 1)
    yy = coords.reshape(1, img_size, 1, 1)
    return mx.sin(6.28 * freq * (xx + yy) + phase)

def main():
    enc, dec = Encoder(), Decoder()
    img = mx.random.normal((2, 64, 64, 3))  # (B, H, W, C)
    z = enc(img)
    out = dec(z)
    mx.eval(z, out)
    assert z.shape == (2, 1024) 
    assert out.shape == (2, 64, 64, 3)

    mx.random.seed(0)
 
    # Config
    batch_size = 16
    img_size = 64          # 64 -> 4 after 4 stride-2 layers
    channels = 3
    latent_dim = 128
    num_steps = 2000
    beta_target = 1e-2
    warmup_steps = 50
 
    model = VAE(in_channels=channels, out_channels=channels, latent_dim=latent_dim)
    mx.eval(model.parameters())
 
    #optimizer = optim.Adam(learning_rate=1e-3)
    optimizer = optim.Adam(learning_rate=optim.cosine_decay(1e-3, num_steps))
    #lr = optim.cosine_decay(1e-4, num_steps)
    #optimizer = optim.Lion(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
 
    for step in range(1, num_steps + 1):
        # Placeholder data: replace with your real (B, H, W, C) frames
        #x = mx.random.normal((batch_size, img_size, img_size, channels))
        x = make_batch(batch_size, img_size, channels) 
 
        beta = beta_target * min(1.0, step / warmup_steps)  # KL warmup
        (loss, recon_loss, kl), grads = loss_and_grad(model, x, beta)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
 
        if step % 100 == 0 or step == 1:
            print(
                f"step {step:4d} | loss {loss.item():.4f} | "
                f"recon {recon_loss.item():.4f} | kl {kl.item():.4f} | beta {beta:.4f}"
            )
 
    # Encode a batch to get the (mean, std) the dynamics model will consume
    x = mx.random.normal((batch_size, img_size, img_size, channels))
    mean, std = model.encoder(x)
    z = sample(mean, std)
    recon = model.decoder(z)
    mx.eval(mean, std, z, recon)
    print("mean:", mean.shape, "std:", std.shape, "z:", z.shape, "recon:", recon.shape)
 
    # Save weights
    model.save_weights("vae.safetensors")
 
 
if __name__ == "__main__":
    main()