import torch

from torch import nn
from einops import repeat, pack, unpack
from huggingface_hub import PyTorchModelHubMixin

from .tokgen_utils.convolution import Convolution
from .tokgen_utils.encoders import MultiScaledScalarEncoder, LinearEncoder
from .vit_utils.positional_encoding import PositionalEncoding
from .vit_utils.transformer import Transformer
# NOTE: Fredformer_backbone is only needed when using FDDM.
# Import lazily inside FDDM_Plugin to avoid extra dependencies for inference-only usage.

# ==================================
# ====       Organization:      ====
# ==================================
# ==== class TokenGeneratorUnit ====
# ==== class ViTUnit            ====
# ==== class TSEncoder8M        ====
# ==== class FDDM_Plugin        ====
# ==== class TSEncoder8MWithFDDM   ====
# ==================================


class FDDM_Plugin(nn.Module):
    """Wraps the Fredformer backbone so it can serve as the plug-in branch."""

    def __init__(
        self,
        seq_len,
        num_channels,
        # block_size=8,
        output_feature_dim=256,
        channel_first=False,
        fredformer_config=None,
    ):
        super().__init__()
        # Lazy import to avoid pulling in FDDM deps unless actually used.
        from ..Flayers.Fredformer_backbone import Fredformer_backbone
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.channel_first = channel_first

        default_cfg = dict(
            ablation=0,
            mlp_drop=0.1,
            use_nys=0,
            output=0,
            cf_dim=512,
            cf_depth=6,
            cf_heads=8,
            cf_mlp=512,   # 512 -> 256
            cf_head_dim=32,
            cf_drop=0.1,
            patch_len=48,
            stride=48,
            d_model=512,
            head_dropout=0.1,
            padding_patch='end',
            individual=False,
            revin=True,
            affine=False,
            subtract_last=False,
            target_window=256,
        )
        if fredformer_config:
            default_cfg.update(fredformer_config)

        target_window = default_cfg.pop("target_window", seq_len)
        self.target_window = target_window
        self.fredformer = Fredformer_backbone(
            c_in=self.num_channels,
            context_window=self.seq_len,
            target_window=target_window,
            **default_cfg,
        )
        flattened_dim = self.num_channels * self.target_window
        self.feature_norm = nn.LayerNorm(flattened_dim)
        # self.final_projection = nn.Linear(flattened_dim, output_feature_dim)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError("Expected tensor of shape (B, C, L) or (B, L, C) for FDDM_Plugin")

        if self.channel_first:
            if x.shape[1] != self.num_channels or x.shape[2] != self.seq_len:
                raise ValueError("channel_first input must be shaped (B, num_channels, seq_len)")
            plugin_input = x
        else:
            if x.shape[1] != self.seq_len or x.shape[2] != self.num_channels:
                raise ValueError("channel_last input must be shaped (B, seq_len, num_channels)")
            plugin_input = x.transpose(1, 2).contiguous()

        freq_features = self.fredformer(plugin_input.contiguous())
        batch = freq_features.shape[0]
        freq_features = freq_features.reshape(batch, -1)
        freq_features = self.feature_norm(freq_features)
        # return self.final_projection(freq_features)
        return freq_features


class TokenGeneratorUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, patch_window_size, scalar_scales, hidden_dim_scalar_enc,
                 epsilon_scalar_enc):
        super().__init__()
        self.num_patches = num_patches
        # token generator for time series objects
        num_ts_feats = 2  # original ts + its diff
        kernel_size = patch_window_size + \
            1 if patch_window_size % 2 == 0 else patch_window_size
        self.convs = nn.ModuleList([
            Convolution(kernel_size=kernel_size,
                        out_channels=hidden_dim, dilation=1)
            for i in range(num_ts_feats)
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(normalized_shape=hidden_dim, eps=1e-5)
            for i in range(num_ts_feats)
        ])

        # token generator for scalar statistics
        if scalar_scales is None:
            scalar_scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]
        num_scalar_stats = 2  # mean + std
        self.scalar_encoders = nn.ModuleList([
            MultiScaledScalarEncoder(
                scalar_scales, hidden_dim_scalar_enc, epsilon_scalar_enc)
            for i in range(num_scalar_stats)
        ])

        # final token projector
        self.linear_encoder = LinearEncoder(
            hidden_dim_scalar_enc * num_scalar_stats + hidden_dim * (num_ts_feats), hidden_dim)

        # scales each time-series w.r.t. its mean and std
        self.ts_scaler = lambda x: (
            x - torch.mean(x, axis=2, keepdim=True)) / (torch.std(x, axis=2, keepdim=True) + 1e-5)

    def forward(self, x):
        with torch.no_grad():
            # compute statistics for each patch
            x_patched = x.reshape(x.shape[0], self.num_patches, -1)
            mean_patched = torch.mean(x_patched, axis=-1, keepdim=True)
            std_patched = torch.std(x_patched, axis=-1, keepdim=True)
            statistics = [mean_patched, std_patched]

        # for each encoder output is (batch_size, num_sub_ts, hidden_dim_scalar_enc)
        scalar_embeddings = [self.scalar_encoders[i](
            statistics[i]) for i in range(len(statistics))]

        # apply convolution for original ts and its diff
        ts_var_embeddings = []
        # diff
        with torch.no_grad():
            diff_x = torch.diff(x, n=1, axis=2)
            # pad by zeros to have same dimensionality as x
            diff_x = torch.nn.functional.pad(diff_x, (0, 1))
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[0](self.ts_scaler(diff_x))
        ts_var_embeddings.append(embedding)
        
        # original ts
        # dim(bs, hidden_dim, len_ts-patch_window_size-1)
        embedding = self.convs[1](self.ts_scaler(x))
        ts_var_embeddings.append(embedding)

        # split ts_var_embeddings into patches
        patched_ts_var_embeddings = []
        for i, embedding in enumerate(ts_var_embeddings):
            embedding = self.layer_norms[i](embedding)
            embedding = embedding.reshape(
                embedding.shape[0], self.num_patches, -1, embedding.shape[2])
            embedding = torch.mean(embedding, dim=2)
            patched_ts_var_embeddings.append(embedding)

        # concatenate diff_x, x, mu and std embeddinga and send them to the linear projector
        x_embeddings = torch.cat([
            torch.cat(patched_ts_var_embeddings, dim=-1),
            torch.cat(scalar_embeddings, dim=-1)
        ], dim=-1)
        x_embeddings = self.linear_encoder(x_embeddings)

        return x_embeddings


class ViTUnit(nn.Module):
    def __init__(self, hidden_dim, num_patches, depth, heads, mlp_dim, dim_head, dropout, device):
        super().__init__()
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim, dropout=dropout, max_len=num_patches+1)
        self.cls_token = nn.Parameter(torch.randn(hidden_dim).to(device))
        self.transformer = Transformer(
            hidden_dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        x_embeddings, ps = pack([cls_tokens, x], 'b * d')
        x_embeddings = self.pos_encoder(
            x_embeddings.transpose(0, 1)).transpose(0, 1)
        x_embeddings = self.transformer(x_embeddings)
        cls_tokens, _ = unpack(x_embeddings, ps, 'b * d')
        return cls_tokens.reshape(cls_tokens.shape[0], -1)


class TSEncoder8M(
    nn.Module,
    PyTorchModelHubMixin,
    # optionally, you can add metadata which gets pushed to the model card
    library_name="tsencoder",
    repo_url="https://huggingface.co/paris-noah/TSEncoder-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """
    The architecture of TSEncoder time series foundation model.

    Parameters
    ----------
    seq_len: int, default 512
        The sequence length, i.e., the length of each time series. This model does not support data with non-fixed
        sequence length, please make all the time series to be of a fixed length by resizing or padding.
    hidden_dim: int, default=256
        Size of a patch (token), i.e., what the hidden dimension each patch is projected to. At the same time,
        ``hidden_dim`` corresponds to the dimension of the embedding space.
    num_patches: int, default=32
        Number of patches (tokens).
    scalar_scales: list, default=None
        List of scales used for MultiScaledScalarEncoder in TokenGeneratorUnit. By default, initialized as [1e-4, 1e-3,
        1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4].
    hidden_dim_scalar_enc: int, default=32
        Hidden dimension of a scalar encoder used for MultiScaledScalarEncoder in TokenGeneratorUnit.
    epsilon_scalar_enc: float, default=1.1
        A constant term used to tolerate the computational error in computation of scale weights for
        MultiScaledScalarEncoder in TokenGeneratorUnit.
    transf_depth: int, default=6
        Number of transformer layers used for Transformer in ViTUnit.
    transf_num_heads: int, default=8
        Number of self-attention heads used for Transformer in ViTUnit.
    transf_mlp_dim: int, default=512
        Hidden dimension of the MLP (feed-forward) transformer's part used for Transformer in ViTUnit.
    transf_dim_head: int, default=128
        Hidden dimension of the keys, queries and values used for Transformer in ViTUnit.
    transf_dropout: froat, default=0.1
        Dropout value used for Transformer in ViTUnit.
    device: {'cpu', 'cuda'}, default='cuda'
        On which device the model is located.
    pre_training: bool, default=False
        If True, applies an MLP projector after the ViTUnit, which originally was used to pre-train the model using
        InfoNCE contrastive loss.
    """

    def __init__(self, seq_len=512, hidden_dim=256, num_patches=32, scalar_scales=None, hidden_dim_scalar_enc=32,
                 epsilon_scalar_enc=1.1, transf_depth=6, transf_num_heads=8, transf_mlp_dim=512, transf_dim_head=128,
                 transf_dropout=0.1, device='cuda', pre_training=False):

        super().__init__()
        assert (seq_len % num_patches) == 0, print(
            'Seq_len must be the multiple of num_patches')
        patch_window_size = int(seq_len / num_patches)

        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.scalar_scales = scalar_scales
        self.hidden_dim_scalar_enc = hidden_dim_scalar_enc
        self.epsilon_scalar_enc = epsilon_scalar_enc
        self.seq_len = seq_len
        self.pre_training = pre_training

        self.tokgen_unit = TokenGeneratorUnit(hidden_dim=hidden_dim,
                                              num_patches=num_patches,
                                              patch_window_size=patch_window_size,
                                              scalar_scales=scalar_scales,
                                              hidden_dim_scalar_enc=hidden_dim_scalar_enc,
                                              epsilon_scalar_enc=epsilon_scalar_enc)
        self.vit_unit = ViTUnit(hidden_dim=hidden_dim, num_patches=num_patches, depth=transf_depth,
                                heads=transf_num_heads, mlp_dim=transf_mlp_dim, dim_head=transf_dim_head,
                                dropout=transf_dropout, device=device)

        self.prj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        # 通过tokgen_unit生成输入x的嵌入表示
        x_embeddings = self.tokgen_unit(x)
        # 将嵌入表示送入视觉转换器(vit_unit)进行处理
        vit_out = self.vit_unit(x_embeddings)
        # 根据是否处于预训练阶段来决定输出
        # if self.pre_training:
        #     # 如果是预训练阶段，则通过投影层(prj)处理vit的输出
        #     print("注意：当前TSEncoder8M模型处于预训练阶段，forward输出经过prj投影层处理")
        #     return self.prj(vit_out)
        # else:
            # 如果不是预训练阶段，则直接返回vit的输出
        #print("注意：当前TSEncoder8M模型不处于预训练阶段，forward输出为vit_unit的直接输出")
        return vit_out


class TSEncoder8MWithFDDM(
    nn.Module,
    PyTorchModelHubMixin,
    library_name="tsencoder",
    repo_url="https://huggingface.co/paris-noah/TSEncoder-8M/tree/main",
    pipeline_tag="time-series-foundation-model",
    license="mit",
    tags=["time-series-foundation-model"],
):
    """Composite model that fuses TSEncoder8M embeddings with FDDM frequency cues."""

    def __init__(
        self,
        seq_len=512,
        hidden_dim=256,
        num_patches=32,
        num_channels=1,
        scalar_scales=None,
        hidden_dim_scalar_enc=32,
        epsilon_scalar_enc=1.1,
        transf_depth=6,
        transf_num_heads=8,
        transf_mlp_dim=512,
        transf_dim_head=128,
        transf_dropout=0.1,
        fddm_output_dim=256,
        device='cuda',
        pre_training=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.hidden_dim = hidden_dim
        self.pre_training = pre_training

        self.tsencoder = TSEncoder8M(
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_patches=num_patches,
            scalar_scales=scalar_scales,
            hidden_dim_scalar_enc=hidden_dim_scalar_enc,
            epsilon_scalar_enc=epsilon_scalar_enc,
            transf_depth=transf_depth,
            transf_num_heads=transf_num_heads,
            transf_mlp_dim=transf_mlp_dim,
            transf_dim_head=transf_dim_head,
            transf_dropout=transf_dropout,
            device=device,
            pre_training=pre_training,
        )

        self.fddm_plugin = FDDM_Plugin(
            seq_len=seq_len,
            num_channels=num_channels,
            output_feature_dim=fddm_output_dim,
            channel_first=True,
        )

        #self.fusion_input_dim = self.tsencoder.hidden_dim + fddm_output_dim
        #self.output_dim = fusion_dim if fusion_dim is not None else self.fusion_input_dim
        # self.fusion_head = nn.Sequential(
        #     nn.LayerNorm(self.fusion_input_dim),
        #     nn.Dropout(fusion_dropout),
        #     nn.Linear(self.fusion_input_dim, self.output_dim),
        # )
        print("注意本次模型forward 只有tsencoder_features,没有contact-----------------------------------")
        self.to(device)

    def to(self, device):
        self.device = device
        return super().to(device)

    def forward(self, x):
        tsencoder_features = self.tsencoder(x)
        freq_features = self.fddm_plugin(x)
        #fused = torch.cat([tsencoder_features, freq_features], dim=-1)
        # return self.fusion_head(fused)
        return freq_features
