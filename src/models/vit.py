import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):

    def __init__(self, in_channels=3, patch_size=4, embed_size=768):
        super(PatchEmbedding, self).__init__()
        # what is relationship between embed size and input image size?
        self.patch_size = patch_size
        self.linear_proj = nn.Linear(in_channels*patch_size*patch_size, embed_size)

    def forward(self, x):

        # break up into batches
        b, ch, h, w = x.shape
        x = torch.reshape(x, shape=(b, ch, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size))
        x = x.permute(0, 2, 4, 1, 3, 5)
        x = x.flatten(1, 2)  # [b, num_patches, ch, patch_size, patch_size]
        x = x.flatten(2, 4)  # [b, num_patches, ch*patch_size*patch_size]

        x = self.linear_proj(x)
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