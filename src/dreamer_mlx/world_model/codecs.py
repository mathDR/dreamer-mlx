import mlx.core as mx
from mlx import nn

from dreamer_mlx.utils import symexp, symlog


class ImageEncoder(nn.Module):
    """
    DreamerV3 Convolutional Encoder.
    Downsamples 64x64 images via 4 strided convolutions with LayerNorm + SiLU.
    """

    def __init__(self, in_channels: int = 3, depth: int = 48, kernel_size: int = 4):
        super().__init__()
        # Channel progression: depth * [1, 2, 4, 8] -> e.g., [48, 96, 192, 384]
        multipliers = [1, 2, 4, 8]
        channels = [depth * m for m in multipliers]
        in_dims = [in_channels] + channels[:-1]

        self.convs = []
        self.norms = []

        for c_in, c_out in zip(in_dims, channels):
            # Stride 2 with padding 1 halves spatial dims at each stage:
            # 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
            self.convs.append(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    bias=False,  # Bias is redundant before LayerNorm
                )
            )
            # MLX LayerNorm normalizes over the last axis (Channels in NHWC)
            self.norms.append(nn.LayerNorm(c_out))

    def __call__(self, x: mx.array) -> mx.array:
        # Normalize uint8 [0, 255] images to [-0.5, 0.5]
        if x.dtype == mx.uint8:
            x = x.astype(mx.float32) / 255.0 - 0.5

        for conv, norm in zip(self.convs, self.norms):
            x = nn.silu(norm(conv(x)))

        # Flatten spatial and channel dimensions: (B, H, W, C) -> (B, H * W * C)
        return x.reshape(x.shape[0], -1)


class VectorEncoder(nn.Module):
    """
    DreamerV3 MLP Encoder for low-dimensional state/proprioceptive vectors.
    Applies symlog to inputs followed by Linear + LayerNorm + SiLU blocks.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.linears = []
        self.norms = []

        curr_dim = in_dim
        for _ in range(num_layers):
            self.linears.append(nn.Linear(curr_dim, hidden_dim, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        x = symlog(x.astype(mx.float32))
        for linear, norm in zip(self.linears, self.norms):
            x = nn.silu(norm(linear(x)))
        return x


class DreamerV3Encoder(nn.Module):
    """
    Unified Multi-Modal Encoder combining visual and vector streams.
    Supports arbitrary leading batch/time dimensions (e.g., (B, T, ...)).
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] | None = (64, 64, 3),
        vector_dim: int | None = None,
        cnn_depth: int = 48,
        mlp_units: int = 512,
        mlp_layers: int = 3,
    ):
        super().__init__()
        if image_shape is None and vector_dim is None:
            raise ValueError("Must specify at least one of image_shape or vector_dim.")


        if image_shape is not None:
            self.has_image = True
            self.img_encoder = ImageEncoder(
                in_channels=image_shape[-1],
                depth=cnn_depth,
            )

        if vector_dim is not None:
            self.has_vector = True
            self.vec_encoder = VectorEncoder(
                in_dim=vector_dim,
                hidden_dim=mlp_units,
                num_layers=mlp_layers,
            )

    def __call__(
        self,
        image: mx.array | None = None,
        vector: mx.array | None = None,
    ) -> mx.array:
        features = []

        if self.has_image:
            if image is None:
                raise ValueError("Expected 'image' input, got None.")
            lead_shape = image.shape[:-3]
            flat_img = image.reshape(-1, *image.shape[-3:])
            img_emb = self.img_encoder(flat_img)
            features.append(img_emb.reshape(*lead_shape, -1))

        if self.has_vector:
            if vector is None:
                raise ValueError("Expected 'vector' input, got None.")
            lead_shape = vector.shape[:-1]
            flat_vec = vector.reshape(-1, vector.shape[-1])
            vec_emb = self.vec_encoder(flat_vec)
            features.append(vec_emb.reshape(*lead_shape, -1))

        return mx.concatenate(features, axis=-1) if len(features) > 1 else features[0]


class ImageDecoder(nn.Module):
    """
    DreamerV3 Transposed Convolutional Decoder.
    Projects latent state to 4x4 feature map, then upsamples to 64x64 via
    4 transposed convolutions with LayerNorm + SiLU.
    """

    def __init__(
        self,
        latent_dim: int,
        out_channels: int = 3,
        depth: int = 48,
        kernel_size: int = 4,
    ):
        super().__init__()
        # Mirror of encoder multipliers [1, 2, 4, 8] -> reversed: [8, 4, 2, 1]
        multipliers = [8, 4, 2, 1]
        channels = [depth * m for m in multipliers]

        # Project latent vector to initial 4x4 spatial grid: (B, 4 * 4 * 8 * depth)
        self.spatial_shape = (4, 4, channels[0])
        self.in_proj = nn.Linear(latent_dim, 4 * 4 * channels[0], bias=False)
        self.in_norm = nn.LayerNorm(channels[0])

        # Channel progression: [8d -> 4d -> 2d -> 1d -> out_channels]
        out_dims = channels[1:] + [out_channels]

        self.deconvs = []
        self.norms = []

        for i, (c_in, c_out) in enumerate(zip(channels, out_dims)):
            is_last = i == len(channels) - 1
            # Stride 2, padding 1 doubles spatial dims at each stage:
            # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=c_in,
                    out_channels=c_out,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=1,
                    bias=is_last,  # Only the final output layer uses bias
                )
            )
            if not is_last:
                self.norms.append(nn.LayerNorm(c_out))

    def __call__(self, latent: mx.array) -> mx.array:
        # Project and reshape to (B, 4, 4, 8 * depth)
        x = self.in_proj(latent)
        x = x.reshape(latent.shape[0], *self.spatial_shape)
        x = nn.silu(self.in_norm(x))

        # Upsample through transposed conv blocks
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)
            if i < len(self.norms):
                x = nn.silu(self.norms[i](x))

        # Shift output mean to match [-0.5, 0.5] normalized image range
        return x + 0.5


class VectorDecoder(nn.Module):
    """
    DreamerV3 MLP Decoder for low-dimensional state/proprioceptive vectors.
    Predicts in symlog space and applies symexp to restore original scale.
    """

    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        self.linears = []
        self.norms = []

        curr_dim = latent_dim
        for _ in range(num_layers):
            self.linears.append(nn.Linear(curr_dim, hidden_dim, bias=False))
            self.norms.append(nn.LayerNorm(hidden_dim))
            curr_dim = hidden_dim

        # Final linear head projects to target vector dimension
        self.out_proj = nn.Linear(hidden_dim, out_dim, bias=True)

    def __call__(self, latent: mx.array) -> mx.array:
        x = latent
        for linear, norm in zip(self.linears, self.norms):
            x = nn.silu(norm(linear(x)))
        # Network predicts symlog(target); symexp maps back to target space
        return symexp(self.out_proj(x))


class DreamerV3Decoder(nn.Module):
    """
    Unified Multi-Modal Decoder reconstructing visual and vector observations
    from the RSSM latent state s_t = [h_t, z_t].
    """

    def __init__(
        self,
        latent_dim: int,
        image_shape: tuple[int, int, int] | None = (64, 64, 3),
        vector_dim: int | None = None,
        cnn_depth: int = 48,
        mlp_units: int = 512,
        mlp_layers: int = 3,
    ):
        super().__init__()
        if image_shape is None and vector_dim is None:
            raise ValueError("Must specify at least one of image_shape or vector_dim.")
        
                
        if image_shape is not None:
            self.has_image = True
            self.img_decoder = ImageDecoder(
                latent_dim=latent_dim,
                out_channels=image_shape[-1],
                depth=cnn_depth,
            )

        if vector_dim is not None:
            self.has_vector = True
            self.vec_decoder = VectorDecoder(
                latent_dim=latent_dim,
                out_dim=vector_dim,
                hidden_dim=mlp_units,
                num_layers=mlp_layers,
            )


    def __call__(self, latent: mx.array) -> dict[str, mx.array]:
        lead_shape = latent.shape[:-1]
        flat_latent = latent.reshape(-1, latent.shape[-1])
        reconstructions = {}

        if self.has_image:
            img_rec = self.img_decoder(flat_latent)
            reconstructions["image"] = img_rec.reshape(*lead_shape, *img_rec.shape[-3:])

        if self.has_vector:
            vec_rec = self.vec_decoder(flat_latent)
            reconstructions["vector"] = vec_rec.reshape(*lead_shape, vec_rec.shape[-1])

        return reconstructions

if __name__ == "__main__":
    batch_size, seq_len = 128, 10
    encoder = DreamerV3Encoder(
        image_shape=(64, 64, 3),
        vector_dim=24,
        cnn_depth=48,  # 'S' / 'M' model size scale
        mlp_units=512,
        mlp_layers=3,
    )

    # Simulated replay buffer batch: (Batch, Time, ...)
    obs_img = mx.random.randint(0, 256, shape=(batch_size, seq_len, 64, 64, 3), dtype=mx.uint8)
    obs_vec = mx.random.normal(shape=(batch_size, seq_len, 24))

    embed = encoder(image=obs_img, vector=obs_vec)
    mx.eval(embed)

    # For 64x64 image (4x4x384 = 6144) + 512 vector = 6656
    print("Output embedding shape:", embed.shape)  # (16, 64, 6656)

    latent_dim = 1536

    decoder = DreamerV3Decoder(
        latent_dim=latent_dim,
        image_shape=(64, 64, 3),
        vector_dim=24,
        cnn_depth=48,
        mlp_units=512,
        mlp_layers=3,
    )

    latent_states = mx.random.normal(shape=(batch_size, seq_len, latent_dim))
    preds = decoder(latent_states)
    mx.eval(preds)

    print("Reconstructed image shape:", preds["image"].shape)    # (16, 64, 64, 64, 3)
    print("Reconstructed vector shape:", preds["vector"].shape)  # (16, 64, 24)
