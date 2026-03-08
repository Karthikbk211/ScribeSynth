import torch
import torch.nn as nn
import torchvision.models as models
from network.attention import (TransformerEncoderLayer, TransformerEncoder,
                               TransformerDecoderLayer, TransformerDecoder,
                               PositionalEncoding1D, PositionalEncoding2D)
from network.feature_extractor import resnet18 as resnet18_dilation, LearnableFrequencyFilter
from einops import rearrange


class StyleContentMixer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 return_intermediate_dec=False, normalize_before=True):
        super(StyleContentMixer, self).__init__()

        # Separate encoder layer instances so style and freq branches have independent weights
        style_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                      dropout, activation, normalize_before)
        style_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.style_encoder = TransformerEncoder(
            style_encoder_layer, num_encoder_layers, style_norm)

        freq_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, activation, normalize_before)
        freq_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.freq_encoder = TransformerEncoder(
            freq_encoder_layer, num_encoder_layers, freq_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        freq_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                     dropout, activation, normalize_before)
        freq_decoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.freq_decoder = TransformerDecoder(freq_decoder_layer, num_decoder_layers, freq_decoder_norm,
                                               return_intermediate=return_intermediate_dec)

        self.add_position1D = PositionalEncoding1D(dropout=0.1, dim=d_model)
        self.add_position2D = PositionalEncoding2D(
            dropout=0.1, d_model=d_model)

        self.high_proj = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_proj = nn.Sequential(
            nn.Linear(512, 4096), nn.GELU(), nn.Linear(4096, 256))
        self.low_feature_filter = nn.Sequential(
            nn.Linear(512, 1), nn.Sigmoid())

        # Reset only the transformer/projection parameters — must happen BEFORE
        # pretrained CNN modules are assigned so Xavier init doesn't overwrite them.
        self._reset_parameters()

        # Pretrained CNN encoders assigned after _reset_parameters so their
        # weights are not overwritten by Xavier initialisation above.
        self.style_encoder_cnn = self._init_resnet18()
        self.style_dilation = resnet18_dilation().conv5_x

        self.freq_filter = LearnableFrequencyFilter(in_channels=1)
        self.freq_encoder_cnn = self._init_resnet18()
        self.freq_dilation = resnet18_dilation().conv5_x

        self.content_encoder = nn.Sequential(
            *([nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)] +
              list(models.resnet18(weights='ResNet18_Weights.DEFAULT').children())[1:-2])
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _init_resnet18(self):
        resnet = models.resnet18(weights='ResNet18_Weights.DEFAULT')
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
        resnet.layer4 = nn.Identity()
        resnet.fc = nn.Identity()
        resnet.avgpool = nn.Identity()
        return resnet

    def _extract_style_feature(self, encoder_cnn, dilation, x, position2D, transformer_encoder):
        x = encoder_cnn(x)
        x = rearrange(x, 'n (c h w) -> n c h w', c=256, h=4).contiguous()
        x = dilation(x)
        x = position2D(x)
        x = rearrange(x, 'n c h w -> (h w) n c').contiguous()
        x = transformer_encoder(x)
        return x

    def _get_low_freq_feature(self, style):
        return self._extract_style_feature(
            self.style_encoder_cnn, self.style_dilation,
            style, self.add_position2D, self.style_encoder)

    def _get_high_freq_feature(self, x):
        x = self.freq_filter(x)
        return self._extract_style_feature(
            self.freq_encoder_cnn, self.freq_dilation,
            x, self.add_position2D, self.freq_encoder)

    def forward(self, style, freq_input, content):
        anchor_style = style[:, 0, :, :].clone().unsqueeze(1).contiguous()
        anchor_freq = freq_input[:, 0, :, :].clone().unsqueeze(1).contiguous()
        anchor_high_feat = self._get_high_freq_feature(anchor_freq)
        anchor_high_nce = self.high_proj(anchor_high_feat)
        anchor_high_nce = torch.mean(anchor_high_nce, dim=0)

        pos_style = style[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_freq = freq_input[:, 1, :, :].clone().unsqueeze(1).contiguous()
        pos_high_feat = self._get_high_freq_feature(pos_freq)
        pos_high_nce = self.high_proj(pos_high_feat)
        pos_high_nce = torch.mean(pos_high_nce, dim=0)

        high_nce_emb = torch.stack([anchor_high_nce, pos_high_nce], dim=1)
        high_nce_emb = nn.functional.normalize(high_nce_emb, p=2, dim=2)

        anchor_low_feat = self._get_low_freq_feature(anchor_style)
        anchor_mask = self.low_feature_filter(anchor_low_feat)
        anchor_low_feat = anchor_low_feat * anchor_mask
        anchor_low_nce = self.low_proj(anchor_low_feat)
        anchor_low_nce = torch.mean(anchor_low_nce, dim=0)

        pos_low_feat = self._get_low_freq_feature(pos_style)
        pos_mask = self.low_feature_filter(pos_low_feat)
        pos_low_feat = pos_low_feat * pos_mask
        pos_low_nce = self.low_proj(pos_low_feat)
        pos_low_nce = torch.mean(pos_low_nce, dim=0)

        low_nce_emb = torch.stack([anchor_low_nce, pos_low_nce], dim=1)
        low_nce_emb = nn.functional.normalize(low_nce_emb, p=2, dim=2)

        content = rearrange(content, 'n t h w -> (n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(
            content, '(n t) c h w -> t n (c h w)', n=style.shape[0]).contiguous()
        content = self.add_position1D(content)

        style_hs = self.decoder(content, anchor_low_feat, tgt_mask=None)
        hs = self.freq_decoder(style_hs[0], anchor_high_feat, tgt_mask=None)

        confidence = self.confidence_head(anchor_low_nce)

        return hs[0].permute(1, 0, 2).contiguous(), high_nce_emb, low_nce_emb, confidence

    def generate(self, style, freq_input, content):
        if style.shape[1] == 1:
            anchor_style = style
            anchor_freq = freq_input
        else:
            anchor_style = style[:, 0, :, :].unsqueeze(1).contiguous()
            anchor_freq = freq_input[:, 0, :, :].unsqueeze(1).contiguous()

        anchor_freq = self.freq_filter(anchor_freq)
        anchor_high_feat = self._get_high_freq_feature(anchor_freq)

        anchor_low_feat = self._get_low_freq_feature(anchor_style)
        anchor_mask = self.low_feature_filter(anchor_low_feat)
        anchor_low_feat = anchor_low_feat * anchor_mask

        content = rearrange(content, 'n t h w -> (n t) 1 h w').contiguous()
        content = self.content_encoder(content)
        content = rearrange(
            content, '(n t) c h w -> t n (c h w)', n=style.shape[0]).contiguous()
        content = self.add_position1D(content)

        style_hs = self.decoder(content, anchor_low_feat, tgt_mask=None)
        hs = self.freq_decoder(style_hs[0], anchor_high_feat, tgt_mask=None)

        confidence = self.confidence_head(
            torch.mean(self.low_proj(anchor_low_feat), dim=0))

        return hs[0].permute(1, 0, 2).contiguous(), confidence
