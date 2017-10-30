import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KERAS_BACKEND'] = "tensorflow"
from run_exp import *
from utils import *
dev = get_available_devices()

def truncated_normal(shape, mean=0.0, stddev=1.0):
    x = np.random.normal(0.0, stddev, shape)
    inds = np.where(np.logical_or(x > 2.0*stddev, x < -2.0*stddev))
    for i in range(10):
        x[inds] = np.random.normal(0.0, stddev, x[inds].shape)
        val = x[inds].shape[0]/float(np.product(shape)) 
        inds = np.where(np.logical_or(x > 2.0*stddev, x < -2.0*stddev))
        if np.all(x < 2.0*stddev) and np.all(x > -2.0*stddev):
            return x + mean

def weight_fn(shape, mode="fan_in", scale = 1.0, distribution='normal'):
    
    receptive_field_size = np.prod(shape[:-2])
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
    
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:
        scale /= max(1., float(fan_in + fan_out) / 2)
        
    if distribution == 'normal':
        stddev = np.sqrt(scale)
        return truncated_normal(shape, 0., stddev).astype(np.float32)
    else:
        limit = np.sqrt(3. * scale)
        return np.random.uniform(-limit, limit, shape).astype(np.float32)

def elu(x, alpha=1.):
    """Exponential linear unit.
    # Arguments
        x: A tenor or variable to compute the activation function for.
        alpha: A scalar, slope of positive section.
    # Returns
        A tensor.
    """
    res = tf.nn.elu(x)
    if alpha == 1:
        return res
    else:
        return tf.where(x > 0, res, alpha * res)

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)

def instance_norm(x):
    epsilon = 1e-9

    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)

    return tf.div(x - mean, tf.sqrt(tf.add(var, epsilon)))

def act_fn(x):
    #x = instance_norm(x)
    x = selu(x)
    return x

style_file = 'inputs/allstar.mp3'
content_file = 'inputs/linkinpark.mp3'

run_exp(content=content_file,
            style=style_file,
            outfile='experiment',
            exp_types=["contentenergyW1.0+contentcqtW1.0+styleharmW50.0+styleenergyW50.0+stylecqtW50.0+L1W0.1"],
            content_times=[16, 55],
            style_times=[17, 190],
            n_filters_stft=4096,
            n_filters_mel=1024,
            n_filters_cqt=256,
            save=True,
            k_h=5,
            k_r=50,
            k_resid=25,
            k_cqt_freq=11,
            k_cqt_time=11,
            suppress_output=True,
            equalize_loss_grads=True,
            maxiter=15000,
            n_hop=None,
            fmax=11025.0,
            n_mels=16,
            n_mels_style=512,
            rates_energy=2,
            num_blocks_cqt=3,
            devices=dev,
            final_save_name='7-24-17/allstar-intheend/11_contentcqt-mel_stylestftW50.0-melW50.0-cqtW50.0-2residmel-3residcqt-nocqtpooling-contentinds-last+L1W5.0.wav',
            use_log=True,
            per_layer_rhythm=False,
            factr=1e-9,
            debug=False,
            use_placeholders=False, 
            return_stft=False,
            use_mag=False,
            n_fft=2048,
            sr=22050,
            n_cqt=84,
            weight_fn=weight_fn,
            act_fn=act_fn, 
            ksize=[1,2,1,1], 
            strides=[1,2,1,1], 
            cqt_pooling=False,
        cqt_content_ind=-1,
        energy_content_ind=-1)

