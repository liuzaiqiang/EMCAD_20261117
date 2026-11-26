import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPA(nn.Module):
    """Refine an encoder skip with decoder edges and local connectivity.

    Inputs have identical [B,C,H,W] shapes. Eight-neighbor cosine similarity
    provides a learned feature connectivity prior. Sobel gradients of the
    decoder's channel mean provide the edge cue. Channel attention fuses the
    two cues before a bounded residual modulation of the skip feature.
    """

    _version = 2

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("CEPA channels must be positive")
        hidden = max(8, channels // 4)
        self.skip_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.decoder_projection = nn.Conv2d(channels, hidden, 1, bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden * 2, max(8, hidden // 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(8, hidden // 4), hidden * 2, 1),
            nn.Sigmoid(),
        )
        self.fuse = nn.Conv2d(hidden * 2, channels, 1)
        # Identity at initialization; tanh keeps modulation bounded thereafter.
        self.residual_scale = nn.Parameter(torch.zeros(()))
        self.edge_context_gain = nn.Parameter(torch.zeros(()))
        self.register_buffer(
            "sobel_x",
            torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3),
            persistent=False,
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3),
            persistent=False,
        )

    @staticmethod
    def _connectivity(features):
        # Compute in float32 for stable cosine similarity under AMP.
        normalized = F.normalize(features.float(), p=2, dim=1, eps=1e-6)
        height, width = normalized.shape[-2:]
        padded = F.pad(normalized, (1, 1, 1, 1), mode="replicate")
        similarity = normalized.new_zeros((normalized.shape[0], 1, height, width))
        for dy in range(3):
            for dx in range(3):
                if dy == 1 and dx == 1:
                    continue
                neighbor = padded[:, :, dy:dy + height, dx:dx + width]
                similarity = similarity + (normalized * neighbor).sum(dim=1, keepdim=True)
        return ((similarity / 8.0) + 1.0) * 0.5

    def _edge(self, decoder):
        gray = decoder.float().mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x.float(), padding=1)
        gy = F.conv2d(gray, self.sobel_y.float(), padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        scale = magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return magnitude / scale

    def forward(self, skip, decoder):
        if skip.shape != decoder.shape:
            raise ValueError("CEPA skip and decoder must have the same [B,C,H,W] shape")
        skip_features = self.skip_projection(skip)
        connectivity = self._connectivity(skip_features)
        edge = self._edge(decoder)
        # The global maximum can hide locally prominent weak edges. Adjust
        # the decoder cue softly without making it a veto on skip features.
        edge_float = edge.float()
        local_mean = F.avg_pool2d(edge_float, 3, stride=1, padding=1, count_include_pad=False)
        local_contrast = (edge_float - local_mean) / (edge_float + local_mean).clamp_min(1e-6)
        gain = 0.1 * self.edge_context_gain.tanh()
        edge = (edge_float * (1.0 + gain * local_contrast)).to(edge.dtype)
        cues = torch.cat((
            skip_features * connectivity.to(skip_features.dtype),
            self.decoder_projection(decoder) * edge.to(decoder.dtype),
        ), dim=1)
        cues = cues * self.channel_attention(cues)
        gate = torch.sigmoid(self.fuse(cues))
        return skip + torch.tanh(self.residual_scale) * skip * gate

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + 'edge_context_gain'
        if local_metadata.get('version', 1) < 2 and key not in state_dict:
            state_dict[key] = torch.zeros_like(self.edge_context_gain)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
