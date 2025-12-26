import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class DAttention(nn.Module):
    """Attention-based MIL (AB-MIL) with simple attention head.

    Args:
        in_dim: input feature dimension per instance
        n_classes: number of slide-level classes
        dropout: if True, applies dropout 0.25 after the encoder
        act: 'relu' or 'gelu' for the encoder activation
        survival: if True, uses survival head (unused in classification)
    """
    def __init__(self, in_dim, n_classes, dropout=False, act='relu', survival=False):
        super(DAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.survival = survival

        feat = [nn.Linear(in_dim, self.L)]
        if act.lower() == 'gelu':
            feat += [nn.GELU()]
        else:
            feat += [nn.ReLU()]
        if dropout:
            feat += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*feat)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )
        self.classifier = nn.Linear(self.L * self.K, n_classes)

    def forward(self, x):
        # x: [N, in_dim]
        feature = self.feature(x).squeeze()
        A = self.attention(feature)
        A = torch.transpose(A, -1, -2)  # K x N
        A_raw = A
        A = F.softmax(A, dim=-1)
        M = torch.mm(A, feature)  # K x L
        logits = self.classifier(M)

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, A_raw, {}

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature = self.feature.to(device)
        self.attention = self.attention.to(device)
        self.classifier = self.classifier.to(device)


class GatedAttention(nn.Module):
    """Gated attention MIL variant (not used by default)."""
    def __init__(self, in_dim, n_classes, dropout=False, act='relu', survival=False):
        super(GatedAttention, self).__init__()
        self.L = 512
        self.D = 128
        self.K = 1
        self.survival = survival

        feat = [nn.Linear(in_dim, self.L)]
        if act.lower() == 'gelu':
            feat += [nn.GELU()]
        else:
            feat += [nn.ReLU()]
        if dropout:
            feat += [nn.Dropout(0.25)]
        self.feature = nn.Sequential(*feat)

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)
        self.classifier = nn.Linear(self.L * self.K, n_classes)

    def forward(self, x):
        feature = self.feature(x).squeeze()
        A_V = self.attention_V(feature)
        A_U = self.attention_U(feature)
        A = self.attention_weights(A_V * A_U)  # N x K
        A_raw = A.detach().clone().squeeze(2) if A.dim() == 3 else A.detach().clone()
        A = torch.transpose(A, 1, 0)  # K x N
        A = F.softmax(A, dim=1)
        M = torch.mm(A, feature)
        logits = self.classifier(M)

        if self.survival:
            Y_hat = torch.topk(logits, 1, dim=1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, None, None

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        return logits, Y_prob, Y_hat, None, {}

