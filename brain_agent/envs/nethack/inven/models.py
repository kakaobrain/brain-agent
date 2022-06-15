import torch.nn as nn

class MyEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, use_index_select=True, **kwargs):
        super().__init__(num_embeddings, embedding_dim, **kwargs)

        self._use_index_select = use_index_select

    def forward(self, x):
        if self._use_index_select:
            out = self.weight.index_select(0, x.view(-1))
            return out.view(x.shape + (-1,))
        else:
            return super().forward(x)

class ActionBaseModel(nn.Module):
    def __init__(self, num_action, emb_dim, use_index_select):
        super().__init__()

        self._emb = MyEmbedding(num_action, emb_dim, use_index_select)

    def forward(self, x):
        """
        (B, A) -> (B, H)
        """

        return self._emb(x)