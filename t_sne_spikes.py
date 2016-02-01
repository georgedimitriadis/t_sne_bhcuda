
from os.path import dirname, join
import h5py as h5
import numpy as np
import t_sne_bhcuda.bhtsne_cuda as TSNE
import time


def t_sne_spikes(kwx_file_path, mask_data=False, path_to_save_tmp_data=None, indices_of_spikes_to_tsne=None,
                 use_scikit=False, perplexity=50.0, theta=0.5, learning_rate=200.0, iterations=1000, gpu_mem=0.8,
                 no_dims=2, eta=200, early_exaggeration=4.0, randseed=-1, verbose=True):

    h5file = h5.File(kwx_file_path, mode='r')
    pca_and_masks = np.array(list(h5file['channel_groups/0/features_masks']))
    masks = np.array(pca_and_masks[:, :, 1])
    pca_features = np.array(pca_and_masks[:, :, 0])
    masked_pca_features = pca_features
    if mask_data:
        masked_pca_features = pca_features * masks

    if not indices_of_spikes_to_tsne:
        num_of_spikes = np.size(masked_pca_features, 0)
        indices_of_spikes_to_tsne = range(num_of_spikes)
    data_for_tsne = masked_pca_features[indices_of_spikes_to_tsne, :]

    if not path_to_save_tmp_data:
        if verbose:
            print('The C++ t-sne executable will save data (data.dat and results.data) in \n{}\n'
                  'You might want to change this behaviour by supplying a path_to_save_tmp_data.\n'.
                  format(dirname(kwx_file_path)))
        path_to_save_tmp_data = dirname(kwx_file_path)

    t0 = time.time()
    t_tsne = TSNE.t_sne(data_for_tsne, use_scikit=use_scikit,files_dir=path_to_save_tmp_data,
                        no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta, iterations=iterations,
                        early_exaggeration=early_exaggeration, gpu_mem=gpu_mem, randseed=randseed, verbose=verbose)
    t_tsne = np.transpose(t_tsne)
    t1 = time.time()
    if verbose:
        print("CUDA t-sne took {} seconds, ({} minutes)".format(t1-t0, (t1-t0)/60))

    np.save(join(dirname(kwx_file_path), 't_sne_results.npy'), t_tsne)

    return t_tsne
