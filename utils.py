import os
import tensorflow as tf
import librosa
from IPython.display import Audio, display
import numpy as np
import matplotlib.pyplot as plt
import scipy
from tensorflow.python.client import device_lib

import scipy.fftpack as fft
from librosa import audio
cqt_frequencies, note_to_hz = librosa.time_frequency.cqt_frequencies, librosa.time_frequency.note_to_hz
stft = librosa.spectrum.stft
estimate_tuning = librosa.pitch.estimate_tuning
from librosa import cache
from librosa import filters
from librosa import util

N_FFT = 2048 # N_FFT used

def get_stft_kernels(n_dft):
    ''' This is the tensorflow version of a function created by
    Keunwoo Choi shown here: https://github.com/keunwoochoi/kapre/blob/master/kapre/stft.py 
    
    Return dft kernels for real/imagnary parts assuming
        the input signal is real.
    An asymmetric hann window is used (scipy.signal.hann).
    Parameters
    ----------
    n_dft : int > 0 and power of 2 [scalar]
        Number of dft components.
    keras_ver : string, 'new' or 'old'
        It determines the reshaping strategy.
    Returns
    -------
    dft_real_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    dft_imag_kernels : np.ndarray [shape=(nb_filter, 1, 1, n_win)]
    * nb_filter = n_dft/2 + 1
    * n_win = n_dft
    '''
    assert n_dft > 1 and ((n_dft & (n_dft - 1)) == 0), \
        ('n_dft should be > 1 and power of 2, but n_dft == %d' % n_dft)

    nb_filter = n_dft / 2 + 1

    # prepare DFT filters
    timesteps = np.arange(n_dft)
    w_ks = (2*np.pi/float(n_dft)) * np.arange(n_dft)
    
    grid = np.dot(w_ks.reshape(n_dft, 1), timesteps.reshape(1, n_dft))
    dft_real_kernels = np.cos(grid)
    dft_imag_kernels = np.sin(grid)
    
    # windowing DFT filters
    dft_window = scipy.signal.hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    dft_real_kernels = dft_real_kernels[:nb_filter].transpose()
    dft_imag_kernels = dft_imag_kernels[:nb_filter].transpose()
    dft_real_kernels = dft_real_kernels[:, np.newaxis, np.newaxis, :]
    dft_imag_kernels = dft_imag_kernels[:, np.newaxis, np.newaxis, :]

    dft_real_kernels = dft_real_kernels.astype(np.float32)
    dft_imag_kernels = dft_imag_kernels.astype(np.float32)
    
    return dft_real_kernels, dft_imag_kernels

def read_audio(filename, tstart=0, tstop=0, sr=44100, n_fft=2048, spectrum=False):
    x, fs = librosa.load(filename, sr=sr)
    if tstart != 0 or tstop!=0:
        x = x[tstart*fs:tstop*fs]
    if spectrum:
        S = np.log1p(np.abs(librosa.stft(x, n_fft)))
        return S, fs, x
    else:
        return fs, x
    

def read_complex_audio_spectrum(filename, tstart=0, tstop=0, n_hop=512):
    x, fs = librosa.load(filename)
    if tstart != 0 or tstop!=0:
        x = x[tstart*fs:tstop*fs]
    S = librosa.stft(x, N_FFT, hop_length=n_hop)

    return S, fs, x

def atan2(y, x, epsilon=1.0e-12):
    # Add a small number to all zeros, to avoid division by zero:
    x = tf.where(tf.equal(x, 0.0), x+epsilon, x)
    y = tf.where(tf.equal(y, 0.0), y+epsilon, y)

    angle = tf.where(tf.greater(x,0.0), tf.atan(y/x), tf.zeros_like(x))
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.greater_equal(y,0.0)), tf.atan(y/x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.less(x,0.0),  tf.less(y,0.0)), tf.atan(y/x) - np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.greater(y,0.0)), 0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.less(y,0.0)), -0.5*np.pi * tf.ones_like(x), angle)
    angle = tf.where(tf.logical_and(tf.equal(x,0.0), tf.equal(y,0.0)), tf.zeros_like(x), angle)
    return angle

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    devices = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if devices == []:
        return ['/cpu:0']
    else:
        return devices

def conv1d(x,
           num_filters,
           filter_length,
           name,
           dilation=1,
           causal=True,
           kernel_initializer=tf.uniform_unit_scaling_initializer(1.0),
           biases_initializer=tf.constant_initializer(0.0),
           kernel=None):
    """ Taken from Magenta's NSynth Wavenet Model
    Fast 1D convolution that supports causal padding and dilation.
    Args:
    x: The [mb, time, channels] float tensor that we convolve.
    num_filters: The number of filter maps in the convolution.
    filter_length: The integer length of the filter.
    name: The name of the scope for the variables.
    dilation: The amount of dilation.
    causal: Whether or not this is a causal convolution.
    kernel_initializer: The kernel initialization function.
    biases_initializer: The biases initialization function.
    Returns:
    y: The output of the 1D convolution.
    """
    batch_size, length, num_input_channels = x.get_shape().as_list()
    if length % dilation != 0:
        dilation = 1
        
    kernel_shape = [1, filter_length, num_input_channels, num_filters]
    if kernel == None:
        std = np.sqrt(2) * np.sqrt(2.0 / ((num_input_channels + num_filters) * filter_length))
        kernel = np.random.standard_normal(kernel_shape)*std
        
        return_kernel = True
    else:
        return_kernel = False
        
    kernel_tf = tf.constant(kernel, name=name, dtype='float32')
    strides = [1, 1, 1, 1]
    biases_shape = [num_filters]
    padding = 'VALID' if causal else 'SAME'

    x_ttb = time_to_batch(x, dilation)
    if filter_length > 1 and causal:
        x_ttb = tf.pad(x_ttb, [[0, 0], [filter_length - 1, 0], [0, 0]])

    x_ttb_shape = x_ttb.get_shape().as_list()
    x_4d = tf.reshape(x_ttb, [x_ttb_shape[0], 1,
                            x_ttb_shape[1], num_input_channels])
    y = tf.nn.conv2d(x_4d, kernel_tf, strides, padding=padding)
    y_shape = y.get_shape().as_list()
    y = tf.reshape(y, [y_shape[0], y_shape[2], num_filters])
    y = batch_to_time(y, dilation)
    y.set_shape([batch_size, length, num_filters])

    return y

def get_logmagnitude_STFT(x_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop):
    
    STFT_real = tf.nn.conv2d(x_,
                            dft_real_kernels_tf,
                            strides=[1, n_hop, 1, 1],
                            padding="SAME",
                            name="conv_dft_real")
    
    STFT_imag = tf.nn.conv2d(x_,
                        dft_imag_kernels_tf,
                        strides=[1, n_hop, 1, 1],
                        padding="SAME",
                        name="conv_dft_imag")
    
    STFT_phase = atan2(STFT_imag, STFT_real)
    STFT_magnitude = tf.sqrt(tf.square(STFT_imag)+tf.square(STFT_real))
    STFT_magnitude = tf.transpose(STFT_magnitude, (0,2,1,3))
    
    STFT_logmagnitude = tf.log1p(STFT_magnitude)
    return STFT_phase, STFT_magnitude, STFT_logmagnitude


def _mel_frequencies(n_mels=128, fmin=0.0, fmax=11025.0):
    """Compute the center frequencies of mel bands.
    `htk` is removed.
    copied from Librosa
    """
    def _mel_to_hz(mels):
        """Convert mel bin numbers to frequencies
        copied from Librosa
        """
        mels = np.atleast_1d(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlfinear scale
        min_log_hz = 1000.0                         # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region
        log_t = (mels >= min_log_mel)

        freqs[log_t] = min_log_hz \
                       * np.exp(logstep * (mels[log_t] - min_log_mel))

        return freqs

    def _hz_to_mel(frequencies):
        """Convert Hz to Mels
        copied from Librosa
        """
        frequencies = np.atleast_1d(frequencies)

        # Fill in the linear part
        f_min = 0.0
        f_sp = 200.0 / 3

        mels = (frequencies - f_min) / f_sp

        # Fill in the log-scale part
        min_log_hz = 1000.0                         # beginning of log region
        min_log_mel = (min_log_hz - f_min) / f_sp   # same (Mels)
        logstep = np.log(6.4) / 27.0                # step size for log region

        log_t = (frequencies >= min_log_hz)
        mels[log_t] = min_log_mel \
                      + np.log(frequencies[log_t] / min_log_hz) / logstep

        return mels

    ''' mel_frequencies body starts '''
    # 'Center freqs' of mel bands - uniformly spaced between limits
    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)

    mels = np.linspace(min_mel, max_mel, n_mels)

    return _mel_to_hz(mels)


def _dft_frequencies(sr=22050, n_dft=2048):
    '''Alternative implementation of `np.fft.fftfreqs` (said Librosa)
    copied from Librosa
    '''
    return np.linspace(0,
                       float(sr) / 2,
                       int(1 + n_dft//2),
                       endpoint=True)


def mel(sr, n_dft, n_mels=128, fmin=0.0, fmax=None):
    ''' create a filterbank matrix to combine stft bins into mel-frequency bins
    use Slaney
    copied from Librosa, librosa.filters.mel
    
    n_mels: numbre of mel bands
    fmin : lowest frequency [Hz]
    fmax : highest frequency [Hz]
        If `None`, use `sr / 2.0`
    '''
    if fmax is None:
        fmax = float(sr) / 2

    # init
    n_mels = int(n_mels)
    weights = np.zeros((n_mels, int(1 + n_dft // 2)))

    # center freqs of each FFT bin
    dftfreqs = _dft_frequencies(sr=sr, n_dft=n_dft)

    # centre freqs of mel bands
    freqs = _mel_frequencies(n_mels + 2,
                             fmin=fmin,
                             fmax=fmax)
    # Slaney-style mel is scaled to be approx constant energy per channel
    enorm = 2.0 / (freqs[2:n_mels+2] - freqs[:n_mels])
    enorm = np.ones(enorm.shape)
    for i in range(n_mels):
        # lower and upper slopes qfor all bins
        lower = (dftfreqs - freqs[i]) / (freqs[i + 1] - freqs[i])
        upper = (freqs[i + 2] - dftfreqs) / (freqs[i + 2] - freqs[i + 1])

        # .. then intersect them with each other and zero
        weights[i] = np.maximum(0, np.minimum(lower, upper)) * enorm[i]

    return weights

def get_variables(y, ndft_else,
                   cqt_name="x", sr=22050, n_hop=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):
    if fmin is None:
        # C1 by default
        fmin = librosa.time_frequency.note_to_hz('C1')

    if tuning is None:
        tuning = estimate_tuning(y=y, sr=sr)

    fft_basis, n_fft, _ = __cqt_filter_fft(sr, fmin, n_bins,
                                           bins_per_octave,
                                           tuning, filter_scale,
                                           norm, sparsity,
                                           hop_length=n_hop,
                                           window=window)

    fft_basis = np.abs(fft_basis).astype('float32').todense()
    fft_basis_tf = tf.constant(fft_basis, name="fft_basis_"+cqt_name, dtype='float32')

    if n_fft == ndft_else:
        dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf = None, None
    else:
        dft_real_kernels_cqt, dft_imag_kernels_cqt = get_stft_kernels(n_fft)
        dft_real_kernels_cqt_tf = tf.constant(dft_real_kernels_cqt, name="dft_real_kernels_cqt_"+cqt_name, dtype='float32')
        dft_imag_kernels_cqt_tf = tf.constant(dft_imag_kernels_cqt, name="dft_imag_kernels_cqt_"+cqt_name, dtype='float32')

    if not scale:
        lengths = filters.constant_q_lengths(sr, fmin,
                                             n_bins=n_bins,
                                             bins_per_octave=bins_per_octave,
                                             tuning=tuning,
                                             window=window,
                                             filter_scale=filter_scale)
        lengths = np.sqrt(lengths[:, np.newaxis] / n_fft).astype('float32')
        lengths_tf = tf.constant(lengths, name="lengths_"+cqt_name, dtype='float32')
    else:
        lengths_tf = None
        
    return dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf, fft_basis_tf, lengths_tf

def get_pseudo_cqt(x_,
                   STFT_magnitude,
                   fft_basis_tf,
                   lengths_tf,
                   cqt_name="x", sr=22050, n_hop=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.00, window='hann', scale=True,
               pad_mode='reflect',
                dft_kernels=None):
    
    if dft_kernels != None:
        dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf = dft_kernels
        _, STFT_magnitude, _ = get_logmagnitude_STFT(x_, dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf, n_hop)
        
    D = tf.transpose(tf.squeeze(STFT_magnitude), (1,0))
    C = tf.matmul(fft_basis_tf, D)

    if scale:
        C /= np.sqrt(n_fft)
    else:
        C = tf.multiply(C, lengths_tf) 
    #C = tf.expand_dims(tf.expand_dims(tf.transpose(C, (1,0)), axis=0), axis=0)
    C = tf.expand_dims(tf.expand_dims(C, axis=0), axis=-1)
    return C

"""
def get_pseudo_cqt(x_, y, 
                   dft_real_kernels_cqt_tf,
                   dft_imag_kernels_cqt_tf,
                   fft_basis_tf,
                   lengths_tf,
                   cqt_name="x", sr=22050, n_hop=512, fmin=None, n_bins=84,
               bins_per_octave=12, tuning=0.0, filter_scale=1,
               norm=1, sparsity=0.01, window='hann', scale=True,
               pad_mode='reflect'):

    _, STFT_magnitude, _ = get_logmagnitude_STFT(x_, dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf, n_hop)

    D = tf.transpose(tf.squeeze(STFT_magnitude), (1,0))
    C = tf.matmul(fft_basis_tf, D)

    if scale:
        C /= np.sqrt(n_fft)
    else:
        C = tf.multiply(C, lengths_tf) 
    C = tf.expand_dims(tf.expand_dims(tf.transpose(C, (1,0)), axis=0), axis=0)
    return C"""



def __cqt_filter_fft(sr, fmin, n_bins, bins_per_octave, tuning,
                     filter_scale, norm, sparsity, hop_length=None,
                     window='hann'):
    '''Generate the frequency domain constant-Q filter basis.'''

    basis, lengths = filters.constant_q(sr,
                                        fmin=fmin,
                                        n_bins=n_bins,
                                        bins_per_octave=bins_per_octave,
                                        tuning=tuning,
                                        filter_scale=filter_scale,
                                        norm=norm,
                                        pad_fft=True,
                                        window=window)

    # Filters are padded up to the nearest integral power of 2
    n_fft = basis.shape[1]

    if (hop_length is not None and
            n_fft < 2.0**(1 + np.ceil(np.log2(hop_length)))):

        n_fft = int(2.0 ** (1 + np.ceil(np.log2(hop_length))))

    # re-normalize bases with respect to the FFT window length
    basis *= lengths[:, np.newaxis] / float(n_fft)

    # FFT and retain only the non-negative frequencies
    fft_basis = fft.fft(basis, n=n_fft, axis=1)[:, :(n_fft // 2)+1]

    # sparsify the basis
    fft_basis = util.sparsify_rows(fft_basis, quantile=sparsity)

    return fft_basis, n_fft, lengths


def __trim_stack(cqt_resp, n_bins):
    '''Helper function to trim and stack a collection of CQT responses'''

    # cleanup any framing errors at the boundaries
    max_col = min(x.shape[1] for x in cqt_resp)

    cqt_resp = np.vstack([x[:, :max_col] for x in cqt_resp][::-1])

    # Finally, clip out any bottom frequencies that we don't really want
    # Transpose magic here to ensure column-contiguity
    return np.ascontiguousarray(cqt_resp[-n_bins:].T).T


def __cqt_response(y, n_fft, hop_length, fft_basis, mode):
    '''Compute the filter response with a target STFT hop.'''

    # Compute the STFT matrix
    D = stft(y, n_fft=n_fft, hop_length=hop_length, window=np.ones,
             pad_mode=mode)

    # And filter response energy
    return fft_basis.dot(D)


def __early_downsample_count(nyquist, filter_cutoff, hop_length, n_octaves):
    '''Compute the number of early downsampling operations'''

    downsample_count1 = max(0, int(np.ceil(np.log2(audio.BW_FASTEST * nyquist /
                                                   filter_cutoff)) - 1) - 1)

    num_twos = __num_two_factors(hop_length)
    downsample_count2 = max(0, num_twos - n_octaves + 1)

    return min(downsample_count1, downsample_count2)


def __early_downsample(y, sr, hop_length, res_type, n_octaves,
                       nyquist, filter_cutoff, scale):
    '''Perform early downsampling on an audio signal, if it applies.'''

    downsample_count = __early_downsample_count(nyquist, filter_cutoff,
                                                hop_length, n_octaves)

    if downsample_count > 0 and res_type == 'kaiser_fast':
        downsample_factor = 2**(downsample_count)

        hop_length //= downsample_factor

        if len(y) < downsample_factor:
            raise ParameterError('Input signal length={:d} is too short for '
                                 '{:d}-octave CQT'.format(len(y), n_octaves))

        new_sr = sr / float(downsample_factor)
        y = audio.resample(y, sr, new_sr,
                           res_type=res_type,
                           scale=True)

        # If we're not going to length-scale after CQT, we
        # need to compensate for the downsampling factor here
        if not scale:
            y *= np.sqrt(downsample_factor)

        sr = new_sr

    return y, sr, hop_length


def __num_two_factors(x):
    """Return how many times integer x can be evenly divided by 2.

    Returns 0 for non-positive integers.
    """
    if x <= 0:
        return 0
    num_twos = 0
    while x % 2 == 0:
        num_twos += 1
        x //= 2

    return num_twos