# Vendored from Matcha-TTS (MIT, see LICENSE in this directory): matcha/models/components/flow_matching.py.
# Only the constructor of BASECFM is kept; cosyvoice.flow.flow_matching.ConditionalCFM overrides the rest.
from abc import ABC

import torch


class BASECFM(torch.nn.Module, ABC):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=128):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-4)
        self.estimator = None
