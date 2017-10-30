from utils import *
import sys
import scipy

input_dev, mel_dev1, mel_dev2, harm_dev, cqt_dev, output_dev = ["/gpu:"+str(i) for i in [0,0,0,0,0,0]]#[0,1,1,0,1,0]]

def get_reduced_rhythm(x_mag, mel_basis_tf, use_log):
    mel_spec = tf.nn.conv2d(
        x_mag,
        mel_basis_tf,
        strides=[1,1,1,1],
        padding="VALID")
             
    if use_log:
        mel_spec = tf.log1p(mel_spec)
    
    return mel_spec

def compute_style_loss(net, style_net, loss_fn):
    if type(net) == list:
        style_loss = 0
        gram = []
        s_gram = []
        for (n, sn) in zip(net, style_net):
            _, height, width, number = map(lambda i: i.value, n.get_shape())
            _, height_style, width_style, number = map(lambda i: i.value, sn.get_shape())
            
            factor = height*width
            style_factor = height_style*width_style
            
            feats_style = tf.reshape(sn, (-1, number))
            feats = tf.reshape(n, (-1, number))
            
            gram += [tf.matmul(tf.transpose(feats), feats)/factor]
            s_gram += [tf.matmul(tf.transpose(feats_style), feats_style)/style_factor]
            style_loss += loss_fn(gram[-1], s_gram[-1], "style")
    else:
        _, height, width, number = map(lambda i: i.value, net.get_shape())
        _, height_style, width_style, number = map(lambda i: i.value, style_net.get_shape())
        
        factor = height*width
        style_factor = height_style*width_style
        
        feats = tf.reshape(net, (-1, number))
        feats_style = tf.reshape(style_net, (-1, number))

        gram = tf.matmul(tf.transpose(feats), feats)/factor
        style_gram = tf.matmul(tf.transpose(feats_style), feats_style)/style_factor
        style_loss = loss_fn(gram, style_gram, "style")

    return style_loss

def get_rhythm_loss(x_rhythm_4content, x_rhythm_4style, x_rhythm_content, x_rhythm_style, rates_energy, kernel_energy_tf, kernel_energy_style_tf, kernels_energy_tf, k_resid, use_dilation, act_fn, loss_fn, include_energy_content, include_energy_style, use_other_for_style, net_energy_content_ind):
    
    if rates_energy == 0:
        padding = "VALID"
    else:
        padding = "SAME"
    
    content_loss_energy, style_loss_energy = [tf.ones(())]*2
    
    conv_energy = tf.nn.atrous_conv2d(
            x_rhythm_4content, 
            kernel_energy_tf,
            1,
            padding=padding,
            name="conv_energy")
    net_energy = act_fn(conv_energy)
    
    d = net_energy
    d_s = [d]
    for i in range(rates_energy):
        if use_dilation:
            dilation = (2**i)
        else:
            dilation = 1
        d = tf.nn.atrous_conv2d(
                d,
                kernels_energy_tf[2*i],
                dilation,
                padding=padding)
        d = act_fn(d)
        
        net_energy += tf.nn.atrous_conv2d(
                d,
                kernels_energy_tf[2*i+1],
                1,
                padding=padding)
        d = act_fn(net_energy)
        d_s += [d]
    net_energy = d
    
    if use_other_for_style:
        conv_energy_4style = tf.nn.atrous_conv2d(
                x_rhythm_4style, 
                kernel_energy_style_tf,
                1,
                padding=padding,
                name="conv_energy")
        net_energy_4style = act_fn(conv_energy_4style)

        d_4style = net_energy_4style
        d_s_4style = [d_4style]
        for i in range(rates_energy):
            if use_dilation:
                dilation = (2**i)
            else:
                dilation = 1
            d_4style = tf.nn.atrous_conv2d(
                    d_4style,
                    kernels_energy_tf[2*i],
                    dilation,
                    padding=padding)
            d_4style = act_fn(d_4style)

            net_energy_4style += tf.nn.atrous_conv2d(
                    d_4style,
                    kernels_energy_tf[2*i+1],
                    1,
                    padding=padding)
            d_4style = act_fn(net_energy_4style)
            d_s_4style += [d_4style]
        net_energy_4style = d_4style
    else:
        d_s_4style = d_s
        
    if include_energy_style:
        conv_energy_style = tf.nn.atrous_conv2d(
                x_rhythm_style, 
                kernel_energy_style_tf,
                1,
                padding=padding,
                name="conv_energy_style")
        net_energy_style = act_fn(conv_energy_style)
        d_style = net_energy_style
        d_s_style = [d_style]
        for i in range(rates_energy):
            if use_dilation:
                dilation = (2**i)
            else:
                dilation = 1
            d_style = tf.nn.atrous_conv2d(
                    d_style,
                    kernels_energy_tf[2*i],
                    dilation,
                    padding=padding)
            d_style = act_fn(d_style)

            net_energy_style += tf.nn.atrous_conv2d(
                    d_style,
                    kernels_energy_tf[2*i+1],
                    1,
                    padding=padding)
            d_style = act_fn(net_energy_style)
            d_s_style += [d_style]
        net_energy_style = d_style

        style_loss_energy = compute_style_loss(d_s_4style, d_s_style, loss_fn)
    else:
        d_s_style = 0
    
    if include_energy_content:
        conv_energy_content = tf.nn.atrous_conv2d(
                x_rhythm_content, 
                kernel_energy_tf,
                1,
                padding=padding,
                name="conv_energy")
        net_energy_content = act_fn(conv_energy_content)
        d_content = net_energy_content
        d_s_content = [d_content]
        for i in range(rates_energy):
            if use_dilation:
                dilation = (2**i)
            else:
                dilation = 1
            d_content = tf.nn.atrous_conv2d(
                    d_content,
                    kernels_energy_tf[2*i],
                    dilation,
                    padding=padding)
            d_content = act_fn(d_content)

            net_energy_content += tf.nn.atrous_conv2d(
                    d_content,
                    kernels_energy_tf[2*i+1],
                    1,
                    padding=padding)
            d_content = act_fn(net_energy_content)
            d_s_content += [d_content]
        net_energy_content = d_content

        content_loss_energy = loss_fn(d_s[net_energy_content_ind], d_s_content[net_energy_content_ind], "content")
    
    return style_loss_energy, content_loss_energy, d_s, d_s_style

def get_harm_loss(x, x_content, x_style, kernel, name, act_fn, loss_fn, include_harm_content, include_harm_style):
    kernel_tf = tf.constant(kernel, name=name, dtype='float32')
    
    content_loss, style_loss = [tf.ones(())]*2
    
    conv = tf.nn.conv2d(
        x,
        kernel_tf,
        padding="VALID",
        strides=[1,1,1,1],
        name="conv_"+name)
    net = act_fn(conv)
    
    if include_harm_content:
        content_conv = tf.nn.conv2d(
                x_content,
                kernel_tf,
                padding="VALID",
                strides=[1,1,1,1],
                name="conv_content"+name)
        content_net = act_fn(content_conv)
        content_loss = loss_fn(net, content_net, "content")
        
    if include_harm_style:
        style_conv = tf.nn.conv2d(
                x_style,
                kernel_tf,
                padding="VALID",
                strides=[1,1,1,1],
                name="conv_style"+name)
        style_net = act_fn(style_conv)
        style_loss = compute_style_loss(net, style_net, loss_fn)
    
    return content_loss, style_loss, net, style_net

def get_cqt_loss(x, x_content, x_style, kernel, k_cqt_freq, k_cqt_time, n_filters_cqt, name, act_fn, loss_fn, weight_fn, num_blocks_cqt, cqt_content_ind, ksize, strides, cqt_pooling, include_cqt_content, include_cqt_style, cqt_pooling_content):
    kernel_tf = tf.constant(kernel, name=name, dtype='float32')
    
    num_blocks = num_blocks_cqt
    kernels_cqt_tf = []
    for i in range(num_blocks):
        kernels_cqt_tf += [tf.constant(weight_fn((k_cqt_freq, k_cqt_time, n_filters_cqt, n_filters_cqt)),
                              dtype='float32'), tf.constant(weight_fn((1, 1, n_filters_cqt, n_filters_cqt)), dtype='float32')]
    
    if num_blocks_cqt == 0:
        padding = "VALID"
    else:
        padding = "SAME"
    
    content_loss, style_loss = [tf.ones(())]*2
    
    conv = tf.nn.atrous_conv2d(
            x, 
            kernel_tf,
            1,
            padding=padding,
            name="conv_cqt1")
    net_cqt = act_fn(conv)
    if cqt_pooling:
        net_cqt = tf.nn.max_pool(
            net_cqt,
            ksize,
            strides,
            padding="VALID")
    d = net_cqt
    d_s = [d]
    for i in range(num_blocks):
        dilation = (2**i)
        d = tf.nn.atrous_conv2d(
                d,
                kernels_cqt_tf[2*i],
                dilation,
                padding=padding)
        d = act_fn(d)
        
        net_cqt += tf.nn.atrous_conv2d(
                d,
                kernels_cqt_tf[2*i+1],
                1,
                padding=padding)
        d = act_fn(net_cqt)
        d_s += [d]
    net_cqt = d
    if include_cqt_style:
        conv_cqt_style = tf.nn.atrous_conv2d(
                x_style, 
                kernel_tf,
                1,
                padding=padding,
                name="conv_cqt_style")
        net_cqt_style = act_fn(conv_cqt_style)
        if cqt_pooling:
            net_cqt_style = tf.nn.max_pool(
                net_cqt_style,
                ksize,
                strides,
                padding="VALID")
        d_style = net_cqt_style
        d_s_style = [d_style]
        for i in range(num_blocks):
            dilation = (2**i)
            d_style = tf.nn.atrous_conv2d(
                    d_style,
                    kernels_cqt_tf[2*i],
                    dilation,
                    padding=padding)
            d_style = act_fn(d_style)

            net_cqt_style += tf.nn.atrous_conv2d(
                    d_style,
                    kernels_cqt_tf[2*i+1],
                    1,
                    padding=padding)
            d_style = act_fn(net_cqt_style)
            d_s_style += [d_style]
        net_cqt_style = d_style
        
        style_loss = compute_style_loss(d_s, d_s_style, loss_fn)
    
    if include_cqt_content:
        conv_cqt_content = tf.nn.atrous_conv2d(
                x_content, 
                kernel_tf,
                1,
                padding=padding,
                name="conv_cqt_content")
        net_cqt_content = act_fn(conv_cqt_content)
        if cqt_pooling:
            net_cqt_content = tf.nn.max_pool(
                net_cqt_content,
                ksize,
                strides,
                padding="VALID")
        d_content = net_cqt_content
        d_s_content = [d_content]
        for i in range(num_blocks):
            dilation = (2**i)
            d_content = tf.nn.atrous_conv2d(
                    d_content,
                    kernels_cqt_tf[2*i],
                    dilation,
                    padding=padding)
            d_content = act_fn(d_content)

            net_cqt_content += tf.nn.atrous_conv2d(
                    d_content,
                    kernels_cqt_tf[2*i+1],
                    1,
                    padding=padding)
            d_content = act_fn(net_cqt_content)
            d_s_content += [d_content]
        net_cqt_content = d_content
    
        content_loss = loss_fn(d_s[cqt_content_ind], d_s_content[cqt_content_ind], "content")
    
    return content_loss, style_loss

def get_prior_loss(x_signal, length, net_type):
    if net_type == "stft":
        class arg(object):
            def __init__(self):
                self.latent_dim = 512 #2048
                self.init = 107 #59
                self.save_dir = "vae2d_2/"
                self.input_size = "64,64"
                self.use_stft = True
                self.use_selu = True
                self.use_disc = True
                self.normalize =  False
                self.batch_norm = False#True
                self.disc_loss_coeff = 1.0
                self.just_input = False # True
                self.rec_loss_coeff = 10.0 # 1.0
    else:
        class arg(object):
            def __init__(self):
                self.latent_dim = 512 #2048
                self.init = 63 #59
                self.save_dir = "vae2d_2/"
                self.input_size = "64,64"
                self.use_stft = False
                self.use_selu = True
                self.use_disc = True
                self.normalize =  False
                self.batch_norm = False#True
                self.disc_loss_coeff = 1.0
                self.just_input = False # True
                self.rec_loss_coeff = 10.0 # 1.0
            
    args = arg()
    
    save_dir = args.save_dir
    lr = 1e-4
    epoch = 200
    g_iter = 1
    d_iter = 1
    latent_dim = args.latent_dim
    if args.use_stft:
        n_dft = 128#512
    else:
        n_dft = 1024
    n_hop = n_dft/4
    n_channels_out = n_dft/2 + 1
    if args.input_size == "64,96":
        n_mels = 96
        n_samples = int(0.74*22050)
    elif args.input_size == "64,64":
        n_samples = int(0.74*22050)
        n_mels = 64

    if args.use_stft:
        n_samples = int(0.74*22050)/8
        n_offset = 10
    
    name = "latent_dim-{}".format(latent_dim)#"{}-{}".format(args.dataset, args.exp_type)
    save_epoch = 1

    if args.rec_loss_coeff != 1.0:
        name += '-rec_loss_coeff-'+str(args.rec_loss_coeff)
    if args.disc_loss_coeff != 1.0:
        name += '-disc_loss_coeff-'+str(args.disc_loss_coeff)
    if args.just_input:
        name += '-just_input'
    if not args.normalize:
        name += '-no-normalization'
    if args.input_size != "64,64":
        name += '-input_size-'+args.input_size
    if args.use_selu:
        name += '-use_selu'

    if not args.batch_norm:
        name += '-no_batchnorm'
    if args.use_stft:
        name += '-use_stft'
        
    input_shape = [None, n_samples, 1, 1]
    
    overflow = length%n_samples
    n_batch = length//n_samples
    if overflow == 0:
        x_signal = tf.reshape(x_signal, [n_batch, n_samples, 1, 1])
    else:
        x_signal_list = []
        for i in range(0, length, n_samples/4):
            if (length - i) >= n_samples:
                x_signal_list += [x_signal[:, i:i+n_samples, :, :]]
            else:
                x_signal_list += [x_signal[:, -n_samples:, :, :]]
        x_signal = tf.concat(x_signal_list, axis=0)
        """
        x_signal_begin = tf.reshape(x_signal[0, :-overflow], [n_batch, n_samples, 1, 1])
        x_signal_end = tf.reshape(x_signal[0, -n_samples:], [1, n_samples, 1, 1])
        x_signal = tf.concat([x_signal_begin, x_signal_end], axis=0)"""
                
    use_noise = 0.0 # variable that lets us test without adding any noise like training for vae.

    initializer = tf.truncated_normal_initializer(stddev=0.02)
    
    if args.batch_norm:
        bn = tf.contrib.layers.batch_norm
    else:
        bn = None

    # Leaky ReLU activation with 0.2 slope as default
    def lrelu(x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
    
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
    
    if args.use_selu:
        act_fn = selu
    else:
        act_fn = lrelu
        
    if args.use_stft:
        scope_string = ""
    else:
        scope_string = "mel"

    def generator(z, reuse=False):
        # generative model / decoder in VAE
        with tf.variable_scope("generator"+scope_string):
            z_feed = z

            if args.input_size == "64,96":
                m, n = 8, 12
            elif args.input_size == "64,64":
                m, n = 8, 8

            if args.use_stft:
                m, n = 8, 9 #4, 33
            nf = 256
            padding = "SAME"
            fc1 = tf.contrib.layers.fully_connected(inputs=z_feed, num_outputs=nf*m*n, reuse=reuse, activation_fn=act_fn, \
                                                    normalizer_fn=bn, \
                                                    weights_initializer=initializer,scope="g_fc1")

            reshaped = tf.reshape(fc1, [-1, m, n, nf])

            k = 5

            conv1 = tf.contrib.layers.conv2d_transpose(reshaped, num_outputs=4*64, kernel_size=k, stride=2, padding=padding,    \
                                            reuse=reuse, activation_fn=act_fn, normalizer_fn=bn, \
                                            weights_initializer=initializer,scope="g_conv1")
            print 'conv1.shape:', conv1.get_shape()

            conv2 = tf.contrib.layers.conv2d_transpose(conv1, num_outputs=2*64, kernel_size=k, stride=2, padding=padding, \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn, \
                                            weights_initializer=initializer,scope="g_conv2")

            print 'conv2.shape:', conv2.get_shape()
            conv3 = tf.contrib.layers.conv2d_transpose(conv2, num_outputs=32, kernel_size=k, stride=2, padding=padding, \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn, \
                                            weights_initializer=initializer,scope="g_conv3")
            print 'conv3.shape:', conv3.get_shape()
            if args.input_size == "64,86":
                conv3 = conv3[:, 4:-5, :, :]
            if args.use_stft:
                conv3 = conv3[:, :, 3:-4, :]
            conv4 = tf.contrib.layers.conv2d(conv3, num_outputs=1, kernel_size=k, stride=1, padding=padding, \
                                            reuse=reuse, activation_fn=gen_act_fn, scope="g_conv4")

            print 'conv4.shape:', conv4.get_shape()
            conv4 = tf.transpose(conv4, (0, 3, 1, 2))
            return conv4

    def discriminator(tensor, z, reuse=False):
        # discriminative model
        with tf.variable_scope("discriminator"+scope_string):
            tensor_t = tf.transpose(tensor, (0, 2, 3, 1))
            k = 5 #5
            nf = 32 # 32
            conv1 = tf.contrib.layers.conv2d(inputs=tensor_t, num_outputs=nf, kernel_size=k, stride=2, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,
                                             weights_initializer=initializer,scope="d_conv1")
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=nf*2, kernel_size=k, stride=2, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,\
                                            weights_initializer=initializer,scope="d_conv2")
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=nf*4, kernel_size=k, stride=2, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,\
                                            weights_initializer=initializer,scope="d_conv3")
            fc1 = tf.contrib.layers.flatten(conv3)
            fc1 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=512, reuse=reuse, activation_fn=act_fn, \
                                                    normalizer_fn=bn, \
                                                    weights_initializer=initializer,scope="d_fc1")
            fc2 = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=1, reuse=reuse, activation_fn=None,\
                                                    weights_initializer=initializer,scope="d_fc2")
            return fc2, conv3

    def inference(tensor, reuse=False):
        # inference model used for vae
        with tf.variable_scope("inference"+scope_string):
            tensor_t = tf.transpose(tensor, (0, 2, 3, 1))
            conv1 = tf.contrib.layers.conv2d(inputs=tensor_t, num_outputs=64, kernel_size=5, stride=2, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn, \
                                             weights_initializer=initializer,scope="i_conv1")
            conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=64, kernel_size=5, stride=2, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,\
                                            weights_initializer=initializer,scope="i_conv2")
            conv3 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=64, kernel_size=5, stride=1, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,\
                                            weights_initializer=initializer,scope="i_conv3")
            conv4 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=64, kernel_size=5, stride=1, padding="SAME", \
                                            reuse=reuse, activation_fn=act_fn,normalizer_fn=bn,\
                                            weights_initializer=initializer,scope="i_conv4")
            fc1 = tf.contrib.layers.flatten(conv4)

            z_mean = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=latent_dim, reuse=reuse, 
                                                       activation_fn=None, scope="z_mean")
            z_log_var = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=latent_dim, reuse=reuse, 
                                                          activation_fn=None, scope="z_log_var")
            return z_mean, z_log_var
        
    dft_real_kernels, dft_imag_kernels = get_stft_kernels(n_dft)
    dft_real_kernels_tf = tf.constant(dft_real_kernels, name="dft_real_kernels", dtype='float32')
    dft_imag_kernels_tf = tf.constant(dft_imag_kernels, name="dft_real_kernels", dtype='float32')

    _, x_mag_energy, x_stft_exact = get_logmagnitude_STFT(x_signal, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop)
    x_stft = x_stft_exact#/tf.reduce_max(x_stft_exact)
    print 'x_stft.shape:', x_stft.get_shape()

    mel_basis = mel(22050, n_dft, n_mels=n_mels)
    mel_basis = np.transpose(mel_basis, (1, 0))
    mel_basis = mel_basis[np.newaxis, np.newaxis, :, :]

    mel_basis_tf = tf.constant(mel_basis.astype('float32'), name="kernel_mel_tf", dtype='float32')
    x_mel = get_reduced_rhythm(x_mag_energy, mel_basis_tf, use_log=True)
    if args.normalize:
        x_mel_normed = x_mel / (tf.reduce_max(x_mel, axis=[2,3], keep_dims=True)+1e-8)#tf.norm(x_mel, axis=[2,3], ord=np.inf, keep_dims=True)
    else:
        x_mel_normed = x_mel / 6.0
    
    if args.use_stft:
        x_mel_normed = x_stft_exact
        #x_mel_normed = x_stft_exact / (tf.reduce_max(x_stft_exact, axis=[2,3], keep_dims=True)+1e-8)
        #x_mel_normed = 2.0*x_mel_normed - 1.0
        gen_act_fn = tf.nn.relu
    else:
        x_mel_normed = 2*x_mel_normed - 1.0
        gen_act_fn = tf.nn.tanh
        
    print 'x_mel.shape:', x_mel_normed.get_shape()
    
    if args.use_disc:
        # just pure GAN stuff
        z_mean, z_log_var = inference(x_mel_normed) # get distribution from inference part of VAE
        z_inf = z_mean # sample from distribution

        x_signal_rec = generator(z_inf)

        d_out_real, dis_layer_orig = discriminator(x_mel_normed, z_inf) 
        d_out_fake_rec, dis_layer_rec = discriminator(x_signal_rec, z_inf, reuse=True)# was z_inf_g

        def reduce_sum(error_tensor):
            # sum over pixel axes, mean over the batch axis
            if len(error_tensor.get_shape()) == 4: # conv representation, reduce channel and first spatial axes first
                error_tensor = tf.reduce_sum(error_tensor, axis=-1)
                error_tensor = tf.reduce_sum(error_tensor, axis=-1)
            error_tensor = tf.reduce_sum(error_tensor, axis=-1)
            return tf.reduce_mean(error_tensor)
        
        #rec_loss = tf.reduce_mean(tf.square(d_out_real-1))#reduce_sum(tf.square(x_mel_normed - x_signal_rec))#reduce_sum(tf.square(dis_layer_orig - dis_layer_rec))
        rec_loss = reduce_sum(tf.square(dis_layer_orig - dis_layer_rec))
        # Get loss terms
        disc_loss = tf.reduce_mean(tf.square(d_out_real-1))
        #disc_loss = tf.reduce_mean(tf.concat([tf.square(d_out_real-1),tf.square(d_out_fake_rec)], axis=-1))
        inf_loss = tf.reduce_mean(tf.square(d_out_fake_rec- 1))

        vae_loss = args.disc_loss_coeff*inf_loss + args.rec_loss_coeff*rec_loss

        kl_loss = 1.0 * - 0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1))
        vae_loss += kl_loss

        gen_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="generator") 
        dis_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator") 
        inf_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="inference") 
        vae_variables = gen_variables + inf_variables

        saver = tf.train.Saver(gen_variables + dis_variables + inf_variables, max_to_keep=200)
        restore_name = '{}vae_gan_saved-{}-{}'.format(args.save_dir+"save/", name, args.init)
    return rec_loss, disc_loss, saver, restore_name

def run_exp(content,
            style,
            outfile,
            exp_types=[],
            n_fft=2048,
            style_times=[0,10],
            content_times=[0,10],
            style_rhythm_times=[0,10],
            n_filters_stft=4096,
            n_filters_mel=4096,
            n_filters_cqt=4096,
            cqt_content_ind=-1, 
            save=False,
            devices="default",
            k_h=11,
            k_r=20,
            k_rhythm=40,
            k_resid=25,
            k_cqt_freq=11,
            k_cqt_time=3,
            num_blocks_cqt=0,
            suppress_output=True,
            griffin_lim=False,
            equalize_loss_grads=True,
            maxiter=300,
            save_flag="",
            n_hop=None,
            n_hop_energy=None,
            n_mels=128,
            n_mels_style=None,
            rates_energy=1,
            num_filters_time=4096,
            filter_length_time=3,
            dilation=2,
            final_save_name=None, 
            fmin=0, 
            fmax="max",
            fmin_cqt=None,
            style_rhythm=None,
            sr=22050,
            use_log=False,
            per_layer_rhythm=False,
            use_dilation=True,
            weight_fn="default", 
            act_fn=tf.nn.relu,
            debug=False, 
            return_stft=False,
            return_feats=False,
            factr=1e7,
            use_placeholders=False, 
            use_mag=False,
            n_cqt=84,
            image_init='',
            loss_fn="L2", 
            ksize=[1,2,1,1], 
            strides=[1,2,1,1], 
            cqt_pooling=False,
            cqt_pooling_content=False,
            energy_content_ind=-1):
    
    include_energy = 'energy' in ''.join(exp_types)
    include_energy_content = 'contentenergy' in ''.join(exp_types)
    include_energy_style = 'styleenergy' in ''.join(exp_types)
    
    include_harm = 'harm' in ''.join(exp_types)
    include_harm_content = 'contentharm' in ''.join(exp_types)
    include_harm_style = 'styleharm' in ''.join(exp_types)
    
    include_cqt = 'cqt' in ''.join(exp_types)
    include_cqt_content = 'contentcqt' in ''.join(exp_types)
    include_cqt_style = 'stylecqt' in ''.join(exp_types)
    
    include_prior_stft = 'priorstft' in ''.join(exp_types)
    include_prior_mel = 'priormel' in ''.join(exp_types)
    
    if final_save_name != None:
        outfile='/'.join(final_save_name.split('/')[:-1])
    
    if not os.path.exists(outfile):
        os.makedirs(outfile)
        
    STYLE_FILENAME = style
    CONTENT_FILENAME = content
    OUTPUT_FILENAME = outfile
    
    if weight_fn == "default":
        def weight_fn(shape):
            std = np.sqrt(2) * np.sqrt(2.0 / ((shape[-2] + shape[-1]) * shape[-3]))
            kernel = np.random.standard_normal(shape)*std
            return kernel.astype('float32')
    
    ALPHA = 1e-2
    
    if loss_fn.upper() == "L2":
        def loss_fn(x, y, type):
            if type == "content":
                return ALPHA * 2 * tf.nn.l2_loss(x - y)
            elif type == "style":
                return 2 * tf.nn.l2_loss(x - y)
    
    if use_placeholders:
        def get_input(input, name):
            return tf.placeholder('float32', (1,len(input),1,1), name=name)
    else:
        def get_input(input, name):
            input_tf = np.ascontiguousarray(input[None,:,None,None])
            return tf.constant(input_tf, name=name, dtype='float32')
        
    fs_content, x_content = read_audio(CONTENT_FILENAME, content_times[0], content_times[1], sr=sr, spectrum=False)
    fs_style, x_style = read_audio(STYLE_FILENAME, style_times[0], style_times[1], sr=sr, spectrum=False)
    
    x_content = x_content/np.abs(x_content).max()
    x_style = x_style/np.abs(x_style).max()
    
    rhythm_flag = False
    if style_rhythm == None:
        rhythm_flag = True
        style_rhythm = style
        style_rhythm_times = style_times
    fs_style_rhythm, x_style_rhythm = read_audio(style_rhythm, style_rhythm_times[0], style_rhythm_times[1], sr=sr, n_fft=n_fft, spectrum=False)
    
    if n_hop == None:
        n_hop = n_fft / 4
    
    if n_hop_energy == None:
        n_hop_energy = n_hop
    
    assert fs_content == fs_style == fs_style_rhythm
    
    fs = fs_content
    
    if fmax == "max":
        fmax = fs/2.0
        
    if not suppress_output:
        print 'Content:', CONTENT_FILENAME, fs_content
        display(Audio(x_content, rate=fs_content))

        print 'Style harm:', STYLE_FILENAME, fs_style
        display(Audio(x_style, rate=fs_style))
        
        print 'Style rhythm:', style_rhythm, fs_style
        display(Audio(x_style_rhythm, rate=fs_style))
        
    dft_real_kernels, dft_imag_kernels = get_stft_kernels(n_fft) # numpy arrays for dft kernels
    
    x_content_tf = np.ascontiguousarray(x_content[None,:,None,None])
    x_style_tf = np.ascontiguousarray(x_style[None,:,None,None])
    x_style_rhythm_tf = np.ascontiguousarray(x_style_rhythm[None,:,None,None])
    
    N_CHANNELS = n_fft // 2 + 1
    # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
    kernel = weight_fn((1, k_h, N_CHANNELS, n_filters_stft))        
    
    input_shape = (1, len(x_content), 1, 1)
    x_size = len(x_content)
    #input_shape_style = (1, len(x_style), 1, 1)
    #input_shape_style_rhythm = (1, len(x_style_rhythm), 1, 1)
    
    if devices == "default":
        devices = get_available_devices()
    print 'Using device(s):', devices
    
    config = tf.ConfigProto(allow_soft_placement = True) #, log_device_placement=True)
    config.gpu_options.allow_growth = True
    
    for exp_type in exp_types:
        print 'Running experiment:', exp_type
        
        if image_init == '':
            image_init = np.random.standard_normal(input_shape).astype(np.float32)*1e-3
        else:
            _, image_init = read_audio(image_init, 0, 100000, sr=sr, spectrum=False)
            image_init = image_init.astype(np.float32).reshape(input_shape)

        result = None
        with tf.Graph().as_default() as g:

            # input
            #for d in devices:
            print 'Loading input vars on gpu 0'
            with g.device(input_dev):
                beta_harm = tf.placeholder(tf.float32, shape=(), name="beta_harm")
                beta_energy = tf.placeholder(tf.float32, shape=(), name="beta_energy")
                beta_cqt = tf.placeholder(tf.float32, shape=(), name="beta_cqt")
                
                # put dft kernels on graph
                dft_real_kernels_tf = tf.constant(dft_real_kernels, name="dft_real_kernels", dtype='float32')
                dft_imag_kernels_tf = tf.constant(dft_imag_kernels, name="dft_imag_kernels", dtype='float32')

                # Define constant inputs
                x_style_ = get_input(x_style, "x_style_")
                _, x_style_mag, x_style = get_logmagnitude_STFT(x_style_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop)

                x_content_ = get_input(x_content, "x_content_")
                _, x_content_mag, x_content = get_logmagnitude_STFT(x_content_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop)

                if rhythm_flag:
                    x_style_rhythm_ = x_style_
                else:
                    x_style_rhythm_ = get_input(x_style_rhythm, "x_content_")
                    _, x_style_rhythm_mag, x_style_rhythm = get_logmagnitude_STFT(x_style_rhythm_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop)

                if n_hop_energy != n_hop:
                    _, x_content_mag_energy, x_content_energy = get_logmagnitude_STFT(x_content_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop_energy)

                    _, x_style_rhythm_mag, x_style_rhythm = get_logmagnitude_STFT(x_style_rhythm_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop_energy)
                else:
                    x_content_mag_energy, x_content_energy = x_content_mag, x_content
                    x_style_rhythm_mag, x_style_rhythm = x_style_mag, x_style
        
                # Define optimizable variable
                x_ = tf.Variable(image_init, name="x_")
                _, x_mag, x = get_logmagnitude_STFT(x_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop)

                if n_hop != n_hop_energy:
                    _, x_mag_energy, x_energy = get_logmagnitude_STFT(x_, dft_real_kernels_tf, dft_imag_kernels_tf, n_hop_energy)
                else:
                    x_mag_energy = x_mag
                    x_energy = x

            # Define mel net
            if include_energy:
                print 'Loading mel on gpu 1'
                with g.device(mel_dev1):
                    mel_basis = mel(fs, n_fft, n_mels=n_mels)
                    mel_basis = np.transpose(mel_basis, (1, 0))
                    mel_basis = mel_basis[np.newaxis, np.newaxis, :, :]

                    mel_basis_tf = tf.constant(mel_basis.astype('float32'), name="kernel_mel_tf", dtype='float32')
                    x_red_4content = get_reduced_rhythm(x_mag_energy, mel_basis_tf, use_log)
                    
                    kernel_energy = weight_fn((1, k_r, n_mels, n_filters_mel))
                    kernel_energy_tf = tf.constant(kernel_energy, name="kernel_energy", dtype='float32')
                    
                    if n_mels_style != None:
                        mel_basis_style = mel(fs, n_fft, n_mels=n_mels_style)
                        mel_basis_style = np.transpose(mel_basis_style, (1, 0))
                        mel_basis_style = mel_basis_style[np.newaxis, np.newaxis, :, :]

                        mel_basis_style_tf = tf.constant(mel_basis_style.astype('float32'), name="kernel_mel_style_tf", dtype='float32')
                        x_red_4style = get_reduced_rhythm(x_mag_energy, mel_basis_style_tf, use_log)
                        use_other_for_style = True
                        
                        kernel_energy_style = weight_fn((1, k_r, n_mels_style, n_filters_mel))
                        kernel_energy_style_tf = tf.constant(kernel_energy_style, name="kernel_energy_style", dtype='float32')
                    else:
                        mel_basis_style_tf = mel_basis_tf
                        x_red_4style = x_red_4content
                        use_other_for_style = False
                        kernel_energy_style_tf = kernel_energy_tf
                    
                    x_red_style = get_reduced_rhythm(x_style_rhythm_mag, mel_basis_style_tf, use_log)
                    x_red_content = get_reduced_rhythm(x_content_mag_energy, mel_basis_tf, use_log)
                
                with g.device(mel_dev2):
                    
                    kernels_energy_tf = []
                    for i in range(rates_energy):
                        kernels_energy_tf += [tf.constant(weight_fn((1, k_resid, n_filters_mel, n_filters_mel)),
                                                          dtype='float32'), tf.constant(weight_fn((1, 1, n_filters_mel, n_filters_mel)), dtype='float32')]

                    style_loss_energy, content_loss_energy, net_energy, net_style_energy = get_rhythm_loss(x_red_4content, x_red_4style, x_red_content, x_red_style, rates_energy,kernel_energy,kernel_energy_style_tf,kernels_energy_tf,  k_resid, use_dilation, act_fn, loss_fn, include_energy_content, include_energy_style, use_other_for_style, energy_content_ind)
                    style_loss_energy = beta_energy * style_loss_energy
                    
                    if include_energy_style:
                        grads_style_energy = tf.gradients(style_loss_energy, x_)[0]
                        grads_style_energy_stft = tf.gradients(style_loss_energy, x_mag_energy)[0]

                        norm_style_energy_stft = tf.norm(grads_style_energy_stft)
                        norm_style_energy = tf.norm(grads_style_energy)
                    else:
                        norm_style_energy = tf.ones(shape=())
                        norm_style_energy_stft = tf.ones(shape=())
                        
                    if include_energy_content:
                        grads_content_energy = tf.gradients(content_loss_energy, x_)[0]
                        grads_content_energy_stft = tf.gradients(content_loss_energy, x_mag_energy)[0]

                        norm_content_energy_stft = tf.norm(grads_content_energy_stft)
                        norm_content_energy = tf.norm(grads_content_energy)
                    else:
                        norm_content_energy = tf.ones(shape=())
                        norm_content_energy_stft = tf.ones(shape=())
                        
            else:
                norm_style_energy = tf.ones(shape=())
                norm_style_energy_stft = tf.ones(shape=())
                norm_content_energy = tf.ones(shape=())
                norm_content_energy_stft = tf.ones(shape=())

            if include_harm:
                print 'Loading STFT on gpu 0'
                with g.device(harm_dev):
                    content_loss, style_loss, net_harm, net_style_harm = get_harm_loss(x, x_content, x_style, kernel, "harm", act_fn, loss_fn, include_harm_content, include_harm_style)
                    style_loss = beta_harm * style_loss
                    
                    if include_harm_style:
                        grads_style_harm = tf.gradients(style_loss, x_)[0]
                        grads_style_harm_stft = tf.gradients(style_loss, x)[0]
                        norm_style_harm_stft = tf.norm(grads_style_harm_stft) 
                        norm_style_harm = tf.norm(grads_style_harm)
                    else:
                        norm_style_harm = tf.ones(shape=())
                        norm_style_harm_stft = tf.ones(shape=())
                        
                    if include_harm_content:
                        grads_content = tf.gradients(content_loss, x_)[0]
                        grads_content_stft = tf.gradients(content_loss, x)[0]
                        norm_content_stft = tf.norm(grads_content_stft)
                        norm_content = tf.norm(grads_content)
                    else:
                        norm_content = tf.ones(shape=())*10
                        norm_content_stft = tf.ones(shape=())*10
                        
            else:
                norm_style_harm = tf.ones(shape=())
                norm_content = tf.ones(shape=())*10
                norm_style_harm_stft = tf.ones(shape=())
                norm_content_stft = tf.ones(shape=())*10

            if include_cqt:
                print 'Loading CQT on gpu 1'
                with g.device(cqt_dev):
                    d = 1
                    b = 12*d
                    f = 0.1/d
                    use_kernels = False

                    dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf, fft_basis_tf, lengths_tf = get_variables(x_content_tf.squeeze(), n_fft, sr=sr, n_hop=n_hop, fmin=None, n_bins=n_cqt, scale=False, filter_scale=f, bins_per_octave=b)

                    if use_kernels:
                        dft_kernels = [dft_real_kernels_cqt_tf, dft_imag_kernels_cqt_tf]
                    else:
                        dft_kernels = None

                    x_cqt = get_pseudo_cqt(x_, x,
                                           fft_basis_tf,
                                           lengths_tf,
                                           sr=sr, n_hop=n_hop, 
                                           fmin=fmin_cqt, n_bins=n_cqt, 
                                           scale=False, filter_scale=f, 
                                           bins_per_octave=12, dft_kernels=dft_kernels)
                    if use_log: x_cqt = tf.log1p(x_cqt)
                    
                    x_content_cqt = get_pseudo_cqt(x_content_, x_content,
                                                   fft_basis_tf,
                                                   lengths_tf,
                                                   sr=sr, n_hop=n_hop, 
                                                   fmin=fmin_cqt, n_bins=n_cqt, 
                                                   scale=False, filter_scale=f, 
                                                   bins_per_octave=b, dft_kernels=dft_kernels)
                    if use_log: x_content_cqt = tf.log1p(x_content_cqt)

                    x_style_cqt = get_pseudo_cqt(x_style_rhythm_, x_style_rhythm,
                                                   fft_basis_tf,
                                                   lengths_tf,
                                                   sr=sr, n_hop=n_hop, 
                                                 fmin=fmin_cqt, n_bins=n_cqt,
                                                 scale=False, filter_scale=f, 
                                                 bins_per_octave=b, dft_kernels=dft_kernels)
                    if use_log: x_style_cqt = tf.log1p(x_style_cqt)

                    #content_loss_cqt = loss_fn(x_cqt, x_content_cqt, "content")

                    kernel_cqt = weight_fn((k_cqt_freq, k_cqt_time, 1, n_filters_cqt))
                    content_loss_cqt, style_loss_cqt = get_cqt_loss(x_cqt, x_content_cqt, x_style_cqt, kernel_cqt, k_cqt_freq, k_cqt_time, n_filters_cqt, "cqt", act_fn, loss_fn, weight_fn, num_blocks_cqt, cqt_content_ind, ksize, strides, cqt_pooling, include_cqt_content, include_cqt_style, cqt_pooling_content)
                    style_loss_cqt = beta_cqt * style_loss_cqt
                    """
                    if cqt_pooling:
                        x_cqt = tf.nn.max_pool(x_cqt,
                                                ksize,
                                                strides,
                                                padding="VALID")

                        x_content_cqt = tf.nn.max_pool(x_content_cqt,
                                                ksize,
                                                strides,
                                                padding="VALID")
                    
                    content_loss_cqt = loss_fn(x_cqt, x_content_cqt, "content")"""

                    if dft_kernels == None:
                        if include_cqt_style:
                            grads_style_cqt_stft = tf.gradients(style_loss_cqt, x)[0]
                            norm_style_cqt_stft = tf.norm(grads_style_cqt_stft) 
                        else:
                            norm_style_cqt_stft = tf.ones(shape=())
                        if include_cqt_content:
                            grads_content_cqt_stft = tf.gradients(content_loss_cqt, x)[0]  
                            norm_content_cqt_stft = tf.norm(grads_content_cqt_stft)
                        else:
                            norm_content_cqt_stft = tf.ones(shape=())*10
                    else:
                        norm_style_cqt_stft = tf.ones(shape=())
                        norm_content_cqt_stft = tf.ones(shape=())*10
                    
                    if include_cqt_style:
                        grads_style_cqt = tf.gradients(style_loss_cqt, x_)[0]
                        norm_style_cqt = tf.norm(grads_style_cqt)
                    else:
                        norm_style_cqt = tf.ones(shape=())
                    
                    if include_cqt_content:
                        grads_content_cqt = tf.gradients(content_loss_cqt, x_)[0] 
                        norm_content_cqt = tf.norm(grads_content_cqt)
                        
                    else:
                        norm_content_cqt = tf.ones(shape=())*10
                        
                    
            else:
                norm_style_cqt = tf.ones(shape=())
                norm_content_cqt = tf.ones(shape=())*10
                norm_style_cqt_stft = tf.ones(shape=())
                norm_content_cqt_stft = tf.ones(shape=())*10
            
            beta_prior = tf.placeholder(shape=[], dtype=tf.float32)
            beta_prior_disc = tf.placeholder(shape=[], dtype=tf.float32)
            
            beta_prior_mel = tf.placeholder(shape=[], dtype=tf.float32)
            beta_prior_disc_mel = tf.placeholder(shape=[], dtype=tf.float32)
            if include_prior_stft:
                rec_loss, disc_loss, saver, restore_name = get_prior_loss(x_, x_size, "stft")
                rec_loss = rec_loss*beta_prior
                disc_loss = disc_loss*beta_prior_disc
            if include_prior_mel:
                rec_loss_mel, disc_loss_mel, saver_mel, restore_name_mel = get_prior_loss(x_, x_size, "mel")
                rec_loss_mel = rec_loss_mel*beta_prior_mel
                disc_loss_mel = disc_loss_mel*beta_prior_disc_mel
            
            with g.device(output_dev):                          
                
                total_style_loss = 0
                total_content_loss = 0
                total_prior_loss = 0
                
                for l_type in exp_type.split('+'):
                    l_type = l_type.split('W')
                    if len(l_type) == 1:
                        weight = 1.0
                    else:
                        weight = float(l_type[1])
                    l_type = l_type[0]
                    if l_type == 'contentharm':
                        print 'adding content harm...'
                        total_content_loss += weight*content_loss
                    elif l_type == 'styleharm':
                        total_style_loss += weight*style_loss
                    elif l_type == 'styleenergy':
                        total_style_loss += weight*style_loss_energy
                    elif l_type == 'contentenergy':
                        total_content_loss += weight*content_loss_energy
                    elif l_type == 'contentcqt' and not include_energy_content:
                        total_content_loss += weight*content_loss_cqt
                    elif l_type == 'contentcqt' and include_energy_content:
                        cqt_weight = weight
                    elif l_type == 'stylecqt':
                        total_content_loss += weight*style_loss_cqt
                    elif l_type == 'L1':
                        total_content_loss += weight*tf.reduce_sum(tf.abs(x))
                    elif l_type == 'priorstft':
                        total_prior_loss += weight*rec_loss
                    elif l_type == 'priorstftdisc':
                        total_prior_loss += weight*disc_loss
                    elif l_type == 'priormel':
                        total_prior_loss += weight*rec_loss_mel
                    elif l_type == 'priormeldisc':
                        total_prior_loss += weight*disc_loss_mel
                    else:
                        print 'WARNING: Loss', l_type, 'not recognized'
                        return 0

                loss =  total_style_loss + total_content_loss + total_prior_loss

                # Optimization
                config = tf.ConfigProto(allow_soft_placement=True)

                with tf.Session(config=config) as sess:
                    sess.run(tf.global_variables_initializer())
                    if include_prior_stft:
                        saver.restore(sess, restore_name)
                        norm_prior = tf.norm(tf.gradients(rec_loss, x_)[0])
                        norm_prior_disc = tf.norm(tf.gradients(disc_loss, x_)[0])
                    else:
                        norm_prior = tf.ones(shape=())
                        norm_prior_disc = tf.ones(shape=())
                        
                    if include_prior_mel:
                        saver_mel.restore(sess, restore_name_mel)
                        norm_prior_mel = tf.norm(tf.gradients(rec_loss_mel, x_)[0])
                        norm_prior_disc_mel = tf.norm(tf.gradients(disc_loss_mel, x_)[0])
                    else:
                        norm_prior_mel = tf.ones(shape=())
                        norm_prior_disc_mel = tf.ones(shape=())
                    
                    feed_dict={beta_harm: 1.0, beta_energy: 1.0, beta_cqt: 1.0,
                                   beta_prior: 1.0, beta_prior_disc: 1.0,
                                   beta_prior_mel: 1.0, beta_prior_disc_mel: 1.0}
                    
                    if use_placeholders:
                        feed_dict[x_content_] = x_content_tf
                        feed_dict[x_style_] = x_style_tf
                        feed_dict[x_style_rhythm_] = x_style_rhythm_tf

                    c = norm_content.eval(feed_dict=feed_dict)
                    s_harm = norm_style_harm.eval(feed_dict=feed_dict)
                    c_energy = norm_content_energy.eval(feed_dict=feed_dict)
                    s_energy = norm_style_energy.eval(feed_dict=feed_dict)
                    c_cqt = norm_content_cqt.eval(feed_dict=feed_dict)
                    s_cqt = norm_style_cqt.eval(feed_dict=feed_dict)
                    
                    p = norm_prior.eval(feed_dict=feed_dict)
                    pd = norm_prior_disc.eval(feed_dict=feed_dict)
                    p_mel = norm_prior_mel.eval(feed_dict=feed_dict)
                    pd_mel = norm_prior_disc_mel.eval(feed_dict=feed_dict)
                    
                    s_harm_stft = norm_style_harm_stft.eval(feed_dict=feed_dict)
                    s_energy_stft = norm_style_energy_stft.eval(feed_dict=feed_dict)
                    s_cqt_stft = norm_style_cqt_stft.eval(feed_dict=feed_dict)
                    c_cqt_stft = norm_content_cqt_stft.eval(feed_dict=feed_dict)

                    print 'Content norm:', c
                    print 'Content CQT norm:', c_cqt
                    print 'Content Mel norm:', c_energy
                    print 'Original style norm:', s_harm
                    print 'Original energy norm:', s_energy

                    print 'Original prior norm:', p
                    print 'Original prior norm-disc:', pd
                    print 'Original prior norm Mel:', p_mel
                    print 'Original prior norm-disc Mel:', pd_mel
                    
                    print 'Content Norm STFT:', norm_content_stft.eval(feed_dict=feed_dict)
                    print 'Content CQT Norm STFT:', c_cqt_stft
                    print 'Style Norm STFT:', norm_style_harm_stft.eval(feed_dict=feed_dict)
                    print 'Style Energy Norm STFT:', norm_style_energy_stft.eval(feed_dict=feed_dict)

                    if debug:
                        returns = []
                        if return_stft:
                            returns += sess.run([x_content, x_style, x], feed_dict=feed_dict)
                        if return_feats:
                            if include_harm:
                                returns += [c.eval(feed_dict=feed_dict) for c in net_style_harm]
                            if include_energy:
                                returns += [c.eval(feed_dict=feed_dict) for c in net_style_energy]
                        return returns

                    if equalize_loss_grads:
                        print 'Normalizing Style Loss Coefficients.'
                        if include_harm_content:
                            beta_vals = [c/s_harm, c/s_energy, c/s_cqt]
                            beta_vals += [p/c, pd/c]
                            beta_vals += [p_mel/c, pd_mel/c]
                            
                        elif include_energy_content:
                            beta_vals = [c_energy/s_harm, c_energy/s_energy, c_energy/s_cqt]#[1.0, s_harm/s_energy] #[c/s_harm, c/s_energy] #s_harm_stft/s_energy_stft]
                            beta_vals += [p/c_energy, pd/c_energy]
                            beta_vals += [p_mel/c_energy, pd_mel/c_energy]
                        elif include_cqt_content:
                            print '\nNormalizing for cqt...\n'
                            beta_vals = [c_cqt/s_harm, c_cqt/s_energy, c_cqt/s_cqt]
                            beta_vals += [p/c_cqt, pd/c_cqt]
                            beta_vals += [p_mel/c_cqt, pd_mel/c_cqt]
                        else:
                            beta_vals = [1.0, s_harm/s_energy, s_harm/s_cqt]
                            beta_vals += [1.0, 1.0]
                            beta_vals += [1.0, 1.0]
                    else:
                        beta_vals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

                    feed_dict[beta_harm] = beta_vals[0]
                    feed_dict[beta_energy] = beta_vals[1]
                    feed_dict[beta_cqt] = beta_vals[2]
                    feed_dict[beta_prior] = beta_vals[3]
                    feed_dict[beta_prior_disc] = beta_vals[4]
                    feed_dict[beta_prior_mel] = beta_vals[5]
                    feed_dict[beta_prior_disc_mel] = beta_vals[6]
                    print('Started optimization.')
                    
                    if include_cqt_content and include_energy_content:
                        #loss -= content_loss_cqt
                        loss += cqt_weight*content_loss_cqt*(c_energy/c_cqt)
                    
                    class Counter(object):
                        def __init__(self):
                            self.iters = 0
                            self.prev_x = image_init.squeeze()

                        def __call__(self, x):
                            #print type(x), x.shape, self.prev_x.shape

                            sys.stdout.write('\riters: {}'.format(self.iters))
                            sys.stdout.flush()
                            #print '{0:4d}    {1:3.6f}'.format(self.iters, self.diff), type(x)
                            self.iters += 1



                            if self.iters % 1000 == 0:
                                result = x_.eval()
                                result = x
                                x_save = result.squeeze()
                                with open(final_save_name[:-4]+'-{}iters.wav'.format(self.iters), 'w+') as f:
                                    scipy.io.wavfile.write(f, fs, x_save)
                                #librosa.output.write_wav(final_save_name[:-4]+'-{}iters.wav'.format(self.iters), x_save, fs, norm=False)
                                if self.iters >= 2000:
                                    try:
                                        os.remove(final_save_name[:-4]+'-{}iters.wav'.format(self.iters-1000))
                                    except:
                                        pass
                    
                    counter = Counter()

                    opt = tf.contrib.opt.ScipyOptimizerInterface(
                      loss, var_list=[x_], method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': factr, 'gtol': factr}, tol=factr)
                   
                    opt.minimize(sess, feed_dict=feed_dict, step_callback=counter)
                    if counter.iters > 1000:
                        try:
                            os.remove(final_save_name[:-4]+'-{}iters.wav'.format((counter.iters//1000)*1000))
                        except:
                            pass

                    print 'Final loss:', loss.eval(feed_dict=feed_dict)
                    result = x_.eval()
                    """
                    if include_cqt:
                        x_cqt_ = x_cqt.eval().squeeze()
                        x_style_cqt_ = x_style_cqt.eval(feed_dict=feed_dict).squeeze()
                        try:
                            fig, ax = plt.subplots(1,2,figsize=(8,6))
                            ax[0].matshow(x_cqt_, aspect=x_cqt_.shape[-1]/float(x_cqt_.shape[0]))
                            ax[0].set_title('Result CQT')
                            ax[1].matshow(x_style_cqt_, aspect=x_style_cqt_.shape[-1]/float(x_style_cqt_.shape[0]))
                            ax[1].set_title('Style CQT')
                            plt.savefig(final_save_name[:-4]+'.png')
                        except:
                            print 'It didn''t work (plotting)... :('"""

        x = result.squeeze()
        
        if save:
            if final_save_name != None:
                total_output_name = final_save_name
            else:
                total_output_name = OUTPUT_FILENAME+exp_type+'_'+save_flag+'.wav'
            x = x/np.abs(x).max()
            librosa.output.write_wav(total_output_name, x, fs, norm=False)
                
        if not suppress_output:
            if final_save_name == None:
                print 'Result for losses:', ', '.join(exp_type.split('+'))
            else:
                print '\n'.join(final_save_name.split('/')[-1].split('_'))
            display(Audio(x, rate=fs))