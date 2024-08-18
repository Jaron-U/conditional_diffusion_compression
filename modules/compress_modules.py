import torch.nn as nn
from .network_components import ResnetBlock, FlexiblePrior, Downsample, Upsample
from .utils import quantize, NormalDistribution


class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super().__init__()
        self.channels = channels # 3
        self.out_channels = out_channels # 64
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)] # [3, 64, 128, 192, 256]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:])) # [(3, 64), (64, 128), (128, 192), (192, 256)]
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels] # [256, 192, 128, 64, 64]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:])) # [(256, 192), (192, 128), (128, 64), (64, 64)]
        assert self.dims[-1] == self.reversed_dims[0]
        self.hyper_dims = [self.dims[-1], *map(lambda m: dim * m, hyper_dims_mults)] # [256, 256, 256, 256]
        self.hyper_in_out = list(zip(self.hyper_dims[:-1], self.hyper_dims[1:])) # [(256, 256), (256, 256), (256, 256)]
        self.reversed_hyper_dims = list(
            reversed([self.dims[-1] * 2, *map(lambda m: dim * m, hyper_dims_mults)])
        ) # [256, 256, 256, 512]
        self.reversed_hyper_in_out = list(
            zip(self.reversed_hyper_dims[:-1], self.reversed_hyper_dims[1:])
        ) # [(256, 256), (256, 256), (256, 512)]
        self.prior = FlexiblePrior(self.hyper_dims[-1])

    def get_extra_loss(self):
        return self.prior.get_extraloss()

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

    def encode(self, input):
        for i, (resnet, down) in enumerate(self.enc):
            input = resnet(input) # ([4, 64, 256, 256])
            input = down(input) # ([4, 64, 128, 128])
        latent = input
        for i, (conv, act) in enumerate(self.hyper_enc):
            input = conv(input)
            input = act(input)
        hyper_latent = input
        q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
        input = q_hyper_latent
        for i, (deconv, act) in enumerate(self.hyper_dec):
            input = deconv(input)
            input = act(input)

        mean, scale = input.chunk(2, 1)
        latent_distribution = NormalDistribution(mean, scale.clamp(min=0.1))
        q_latent = quantize(latent, "dequantize", latent_distribution.mean)        
        state4bpp = {
            "latent": latent,
            "hyper_latent": hyper_latent,
            "latent_distribution": latent_distribution,
        }
        return q_latent, q_hyper_latent, state4bpp

    def decode(self, input):
        output = []
        for i, (resnet, up) in enumerate(self.dec):
            input = resnet(input)
            input = up(input)
            output.append(input)
        return output[::-1]

    def bpp(self, shape, state4bpp):
        B, _, H, W = shape
        latent = state4bpp["latent"]
        hyper_latent = state4bpp["hyper_latent"]
        latent_distribution = state4bpp["latent_distribution"]
        if self.training:
            q_hyper_latent = quantize(hyper_latent, "noise")
            q_latent = quantize(latent, "noise")
        else:
            q_hyper_latent = quantize(hyper_latent, "dequantize", self.prior.medians)
            q_latent = quantize(latent, "dequantize", latent_distribution.mean)
        hyper_rate = -self.prior.likelihood(q_hyper_latent).log2()
        cond_rate = -latent_distribution.likelihood(q_latent).log2()
        temp = hyper_rate.sum(dim=(1, 2, 3))
        temp2 = cond_rate.sum(dim=(1, 2, 3))
        bpp = (hyper_rate.sum(dim=(1, 2, 3)) + cond_rate.sum(dim=(1, 2, 3))) / (H * W)
        return bpp

    def forward(self, input):
        q_latent, q_hyper_latent, state4bpp = self.encode(input)
        bpp = self.bpp(input.shape, state4bpp)
        output = self.decode(q_latent)
        return {
            "output": output,
            "bpp": bpp,
            "q_latent": q_latent,
            "q_hyper_latent": q_hyper_latent,
        }


class ResnetCompressor(Compressor):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        hyper_dims_mults=(4, 4, 4),
        channels=3,
        out_channels=3,
    ):
        super().__init__(
            dim,
            dim_mults,
            reverse_dim_mults,
            hyper_dims_mults,
            channels,
            out_channels
        )
        self.build_network()

    def build_network(self):

        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])
        self.hyper_enc = nn.ModuleList([])
        self.hyper_dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            is_last = ind >= (len(self.in_out) - 1)
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        Upsample(dim_out if not is_last else dim_in, dim_out),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.hyper_in_out):
            is_last = ind >= (len(self.hyper_in_out) - 1)
            self.hyper_enc.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if ind == 0
                        else nn.Conv2d(dim_in, dim_out, 5, 2, 2),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(self.reversed_hyper_in_out):
            is_last = ind >= (len(self.reversed_hyper_in_out) - 1)
            self.hyper_dec.append(
                nn.ModuleList(
                    [
                        nn.Conv2d(dim_in, dim_out, 3, 1, 1)
                        if is_last
                        else nn.ConvTranspose2d(dim_in, dim_out, 5, 2, 2, 1),
                        nn.LeakyReLU(0.2) if not is_last else nn.Identity(),
                    ]
                )
            )
