import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention2(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention2, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K),
        )

    def forward(self, x, isNorm=True):
        A = self.attention(x)
        A = torch.transpose(A, 1, 0)  # K x N
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class Attention_Gated(nn.Module):
    def __init__(self, L=512, D=128, K=1):
        super(Attention_Gated, self).__init__()
        self.L = L
        self.D = D
        self.K = K
        self.attention_V = nn.Sequential(nn.Linear(self.L, self.D), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(self.L, self.D), nn.Sigmoid())
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x, isNorm=True):
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)  # N x K
        A = torch.transpose(A, 1, 0)  # K x N
        if isNorm:
            A = F.softmax(A, dim=1)
        return A


class Classifier_1fc(nn.Module):
    def __init__(self, n_channels, n_classes, droprate=0.0, survival=False):
        super(Classifier_1fc, self).__init__()
        self.fc = nn.Linear(n_channels, n_classes)
        self.droprate = droprate
        self.survival = survival
        if self.droprate != 0.0:
            self.dropout = nn.Dropout(p=self.droprate)

    def forward(self, x):
        if self.droprate != 0.0:
            x = self.dropout(x)
        x = self.fc(x)
        if self.survival:
            Y_hat = torch.topk(x, 1, dim=1)[1]
            hazards = torch.sigmoid(x)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return x


class residual_block(nn.Module):
    def __init__(self, nChn=512):
        super(residual_block, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nChn, nChn, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)


class DimReduction(nn.Module):
    def __init__(self, n_channels, m_dim=512, numLayer_Res=0):
        super(DimReduction, self).__init__()
        self.fc1 = nn.Linear(n_channels, m_dim, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.numRes = numLayer_Res
        res = []
        for _ in range(numLayer_Res):
            res.append(residual_block(m_dim))
        self.resBlocks = nn.Sequential(*res)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        if self.numRes > 0:
            x = self.resBlocks(x)
        return x


class Attention_with_Classifier(nn.Module):
    def __init__(self, L=512, D=128, K=1, num_cls=2, droprate=0.0, survival=False):
        super(Attention_with_Classifier, self).__init__()
        self.attention = Attention_Gated(L, D, K)
        self.classifier = Classifier_1fc(L, num_cls, droprate)
        self.survival = survival

    def forward(self, x):
        AA = self.attention(x)  # K x N
        afeat = torch.mm(AA, x)  # K x L
        pred = self.classifier(afeat)  # K x num_cls
        if self.survival:
            Y_hat = torch.topk(pred, 1, dim=1)[1]
            hazards = torch.sigmoid(pred)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None
        return pred


class DTFD_MIL(nn.Module):
    """DTFD-MIL style model (ported): Dim reduction + gated attention + 1fc classifier.

    This wraps the components to expose a consistent MIL API:
    returns (logits, Y_prob, Y_hat, A_raw/None, results_dict)
    """

    def __init__(self, in_dim, n_classes, dropout=False, m_dim=512, d_attn=128, num_heads=1, survival=False):
        super().__init__()
        self.reducer = DimReduction(n_channels=in_dim, m_dim=m_dim, numLayer_Res=0)
        self.attn_head = Attention_with_Classifier(L=m_dim, D=d_attn, K=num_heads, num_cls=n_classes, droprate=0.25 if dropout else 0.0, survival=survival)
        self.survival = survival

    def forward(self, x):
        # x: [N, in_dim]
        h = self.reducer(x)
        out = self.attn_head(h)
        if self.survival:
            hazards, S, Y_hat, _, _ = out
            return hazards, S, Y_hat, None, None
        logits = out  # [K, n_classes], typically K=1
        if logits.dim() == 2 and logits.shape[0] == 1:
            logits = logits  # [1, C]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, None, {}

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reducer = self.reducer.to(device)
        self.attn_head = self.attn_head.to(device)

