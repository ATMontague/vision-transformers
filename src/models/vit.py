import torch
import torch.nn as nn


class PatchEmbed(nn.Module):

    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=768):
        """
        Split image into patches and then embed them.

        :param img_size: int
            size of the input image (must be square)
        :param patch_size: int
            size of the patches to be created (must be square)
        :param in_channels: int
            num of input channels
        :param embed_dim:
            the embedding dimension (will be constant across entire network)
        """

        super(PatchEmbed).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)
