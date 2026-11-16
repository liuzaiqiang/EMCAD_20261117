import torch

def bidirectional_feature_consistency(decoder, aligned_skip, raw_skip, reverse_decoder):
    """Two stop-gradient directions; normalization prevents scale from dominating."""
    fn = torch.nn.functional.normalize
    d = fn(decoder.float(), dim=1)
    a = fn(aligned_skip.float(), dim=1)
    s = fn(raw_skip.float(), dim=1)
    r = fn(reverse_decoder.float(), dim=1)
    return .5 * ((d - a.detach()).square().sum(dim=1).mean() +
                 (s - r.detach()).square().sum(dim=1).mean())
