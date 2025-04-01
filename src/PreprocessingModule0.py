import torch
# import numpy as np

class PreprocessingModule:
    def __init__(self, correlation_threshold=0.998):
        self.correlation_threshold = correlation_threshold

    def ya_preprocess_cube(self, cube, mod='cross-fuse'):
        batch_size, n_sp, spectral_dim = cube.shape
        selected_spectra = cube
        cross_fused_spectra = torch.empty((batch_size, 4 * spectral_dim), device=cube.device)
        max_spectra = torch.max(selected_spectra[:, :-1, :], dim=1).values
        min_spectra = torch.min(selected_spectra[:, :-1, :], dim=1).values
        self_spectra = selected_spectra[:, -1, :]
        mean_spectra = torch.mean(selected_spectra[:, :-1, :], dim=1)
        if mod == 'cross-fuse':
            cross_fused_spectra[:, 0::4] = min_spectra
            cross_fused_spectra[:, 1::4] = mean_spectra
            cross_fused_spectra[:, 2::4] = self_spectra
            cross_fused_spectra[:, 3::4] = max_spectra  # A[:-2,:,:]
        else:  # mod == 'direct-connect'
            cross_fused_spectra[:, :spectral_dim] = min_spectra
            cross_fused_spectra[:, spectral_dim:spectral_dim*2] = mean_spectra
            cross_fused_spectra[:, spectral_dim*2:-spectral_dim] = self_spectra
            cross_fused_spectra[:, -spectral_dim:] = max_spectra  # [:, -4, :]

        cross_fused_spectra.cuda()
        return cross_fused_spectra
