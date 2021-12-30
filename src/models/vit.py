import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Image to Patches, Linear Projection, and Embedding
    """

    def __init__(self, in_channels=3, image_size=256, patch_size=4, embed_size=768):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.input_size = in_channels*patch_size*patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.linear_proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_size, kernel_size=patch_size,
                                     stride=patch_size)

    def forward(self, x):
        """
        Forward pass.
        :param x: tensor w/ shape: [b, in_channels, h, w]
        :return: tensor w/ shape: [b, num_patches, embed_size]
        """

        x = self.linear_proj(x)  # [b, embed_size, num_patches/2, num_patches/2]
        x = x.flatten(2)  # flatten patches into single dim
        x = x.transpose(1, 2)
        return x


class ViT(nn.Module):

    def __init__(self, input_size, output_size, hidden_dim=128, num_heads=2, dim_feedforward=2048,
                 dim_k=96, dim_v=96, dim_q=96):
        """
        :param input_size: ...
        :param output_size: ...
        :param hidden_dim: the dimensionality of the output embeddings used in the final layer
        :param num_heads: number of transformer heads to use
        :param dim_feedforward: the dimension of the feedforwardward network
        :param dim_k: dimension of key vectors
        :param dim_v: dimension of value vectors
        :param dim_q: dimension of query vectors
        """
        pass

    def embed(self, inputs):
        """
        Convert input images into patches and then linear project them so that they can be fed to the transformer.
        :param inputs: tensor of shape [b, h, w, 3]
        :return:
        """
        pass


if __name__ == '__main__':
    patch = PatchEmbedding()
