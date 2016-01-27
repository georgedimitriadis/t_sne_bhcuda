#!/usr/bin/env python

'''
A simple Python wrapper for the t_sne_bhcuda binary.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

This code is a small extension to the original python wrapper by Pontus Stenetorp
which passes to the t_sne_bhcuda executable the amount of gpu
memory to be used. It also splits the read, write and execute parts into separate
functions that can be used independently.

It also acts as a thin wrapper to the scikit learn t-sne implementation
(which can be called instead of the t_sne_bhcuda executable).

The data into the t_sne function (or the save_data_for_tsne function) is a samples x features array.

Example

import bhtsne_cuda
import matplotlib.pyplot as plt

perplexity = 50.0
theta = 0.5
learning_rate = 200.0
iterations = 2000
gpu_mem = 0.8
t_sne_result = bhtsne_cuda.t_sne(samples=data_for_tsne, tmp_dir_path=r'C:\temp\tsne_results',
                        no_dims=2, perplexity=perplexity, eta=learning_rate, theta=theta,
                        iterations=iterations, gpu_mem=gpu_mem, randseed=-1, verbose=3)
t_sne_result = np.transpose(t_sne_result)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(t_sne_result[0], t_sne_result[1])


Author:     George Dimitriadis    <george dimitriadis uk>
Version:    2016-01-20
'''

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from os.path import abspath, dirname, isfile, join as path_join
from struct import calcsize, pack, unpack
from subprocess import Popen, PIPE
from platform import system
import sys

### Constants
IS_WINDOWS = True if system() == 'Windows' else False
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bin', 'windows', 't_sne_bhcuda.exe') if IS_WINDOWS \
                     else path_join(dirname(__file__), 'bin', 'linux', 't_sne_bhcuda')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the t_sne_bhcuda.exe binary in the '
                                  'same directory as this script, have you forgotten to compile it?: {}'
                                  ).format(BH_TSNE_BIN_PATH)

# Default hyper-parameter values
DEFAULT_NO_DIMS = 2
DEFAULT_PERPLEXITY = 50.0
DEFAULT_EARLY_EXAGGERATION = 4.0
DEFAULT_THETA = 0.5
EMPTY_SEED = -1
DEFAULT_ETA = 200
DEFAULT_ITERATIONS = 500
DEFAULT_GPU_MEM = 0
###


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def t_sne(samples, use_scikit=False, tmp_dir_path=None, results_filename='result.dat', data_filename='data.dat',
          no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA, eta=DEFAULT_ETA,
          iterations=DEFAULT_ITERATIONS, early_exaggeration = DEFAULT_EARLY_EXAGGERATION,
          gpu_mem=DEFAULT_GPU_MEM, randseed=EMPTY_SEED, verbose=False):

    if use_scikit:  # using python's scikit tsne implementation
        from sklearn.manifold import TSNE as tsne
        if DEFAULT_THETA > 0:
            method = 'barnes_hut'
        elif DEFAULT_THETA == 0:
            method = 'exact'
        model = tsne(n_components=2, perplexity=perplexity, early_exaggeration=early_exaggeration,
                     learning_rate=eta, n_iter=DEFAULT_ITERATIONS, n_iter_without_progress=500,
                     min_grad_norm=1e-7, metric="euclidean", init="random", verbose=verbose,
                     random_state=None, method='barnes_hut', angle=theta)
        t_sne_results = model.fit_transform(samples)

        return t_sne_results

    else:  # using the C++/cuda implementation
        save_data_for_tsne(samples, tmp_dir_path, data_filename, theta, perplexity,
                           eta, no_dims, iterations, gpu_mem, randseed)
        # Call t_sne_bhcuda and let it do its thing
        print((abspath(BH_TSNE_BIN_PATH), ))
        with Popen([abspath(BH_TSNE_BIN_PATH), ], cwd=tmp_dir_path, stdout=PIPE, bufsize=1, universal_newlines=True) \
                as t_sne_gpu_p:
            for line in iter(t_sne_gpu_p.stdout):
                print(line, end='')
                sys.stdout.flush()
            t_sne_gpu_p.wait()
        assert not t_sne_gpu_p.returncode, ('ERROR: Call to t_sne_bhcuda exited '
                                            'with a non-zero return code exit status, please ' +
                                            ('enable verbose mode and ' if not verbose else '') +
                                            'refer to the t_sne_bhcuda output for further details')

        return load_tsne_result(tmp_dir_path, results_filename)


def save_data_for_tsne(samples, tmp_dir_path, filename, theta, perplexity, eta, no_dims, iterations, gpu_mem, randseed):
    # Assume that the dimensionality of the first sample is representative for the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    with open(path_join(tmp_dir_path, filename), 'wb') as data_file:
        # Write the t_sne_bhcuda header
        data_file.write(pack('iidddiif', sample_count, sample_dim, theta,
                             perplexity, eta, no_dims, iterations, gpu_mem))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))
        # Write random seed if specified
        if randseed != EMPTY_SEED:
            data_file.write(pack('i', randseed))


def load_tsne_result(tmp_dir_path, filename):
    t_sne_results = []
    # Read and pass on the results
    with open(path_join(tmp_dir_path, filename), 'rb') as output_file:
        # The first two integers are just the number of samples and the dimensionality
        result_samples, result_dims = _read_unpack('ii', output_file)
        # Collect the results, but they may be out of order
        results = [_read_unpack('{}d'.format(result_dims), output_file)
            for _ in range(result_samples)]
        # Now collect the landmark data so that we can return the data in
        #   the order it arrived
        results = [(_read_unpack('i', output_file), e) for e in results]
        # Put the results in order and yield it
        results.sort()
        for _, result in results:
            t_sne_results.append(result)
        return t_sne_results
        # The last piece of data is the cost for each sample, we ignore it
        #read_unpack('{}d'.format(sample_count), output_file)


