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


class AttentionHead(nn.Module):

    def __init__(self, k_dim, v_dim, q_dim, embed_dim):
        super(AttentionHead, self).__init__()
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.q_dim = q_dim
        self.embed_dim = embed_dim

        self.k = nn.Linear(self.embed_dim, self.k_dim)
        self.v = nn.Linear(self.embed_dim, self.v_dim)
        self.q = nn.Linear(self.embed_dim, self.q_dim)

        self.softmax = nn.Softmax(dim=2)  # todo: verify which dim to use
        self.projection = nn.Linear(self.v_dim, self.embed_dim)

    def forward(self, x):

        k = self.k(x)  # [b, embed_size, k_dim] -> [b, 64, 96] 64 is number of patches
        v = self.v(x)  # [b, embed_size, v_dim] -> [b, 64, 96]
        q = self.q(x)  # [b, embed_size, q_dim] -> [b, 64, 96]

        print('k: ', k.shape)
        print('v: ', v.shape)
        print('q: ', q.shape)

        # multiply keys and queries

        temp = torch.bmm(input=q, mat2=k.permute(0, 2, 1)) / (self.k_dim ** 0.5)  # [b, embed_size, k_dim]
        print('matmul of keys & queries: ', temp.shape)

        # softmax to norm vals to be between 0 and 1
        temp = self.softmax(temp)

        # multiply with values to get final result
        output = torch.bmm(input=temp, mat2=v)

        # ensure that output shape is same as input shape
        output = self.projection(output)

        # we should have updated vectors here. return...?
        return output


class ViT(nn.Module):

    def __init__(self, image_size, patch_size, embed_size=768, num_heads=2, dim_feedforward=2048,
                 k_dim=96, v_dim=96, q_dim=96):
        super(ViT, self).__init__()

        self.embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, embed_size=embed_size)
        self.attention = AttentionHead(k_dim=k_dim, v_dim=v_dim, q_dim=q_dim, embed_dim=embed_size)

    def forward(self, x):
        """
        :param x: tensor w/ shape [b, 3, h, w]
        :return: tensor
        """

        # patch embed input images
        x = self.embed(x)  # [b, num_patches, embed_size]
        print('after embedding: ', x.shape)

        # pass through attention head
        x = self.attention(x)  # [b, num_patches, embed_size]
        print('after attention: ', x.shape)
        return x
