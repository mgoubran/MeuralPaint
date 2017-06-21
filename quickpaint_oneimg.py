#!/usr/bin/env python
# Maged Goubran @ 2016, mgoubran@stanford.edu

# coding: utf-8

import transform
import numpy as np
import os
import tensorflow as tf
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import defaultdict
from scipy.misc import imread, imsave

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # filter out info & warning logs


# read input arguments
def get_opts():
    parser = ArgumentParser(description="Paint (transfer style to) image using a pre-trained neural network model.",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('-m', '--model', type=str,
                        dest='model_dir',
                        help='dir or .ckpt file to load model from',
                        metavar='MODEL', required=True)
    parser.add_argument('-i', '--input', type=str,
                        dest='in_path', help='dir or file to transform (content)',
                        metavar='IN_PATH', required=True)
    parser.add_argument('-o', '--output', type=str,
                        dest='out_path', help='destination (dir or file) of transformed input (stylized content)',
                        metavar='OUT_PATH', required=True)
    parser.add_argument('-d', '--device', type=str,
                        dest='device', help='device to perform compute on',
                        metavar='', default='/gpu:0')
    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for feedforwarding',
                        metavar='', default=4)

    opts = parser.parse_args()

    # check inputs
    assert os.path.exists(opts.in_path), 'Input dir: %s does not exist!' % opts.in_path

    if os.path.isdir(opts.out_path):
        if not os.path.exists(opts.out_path):
            print('creating output dir')
            os.makedirs(opts.out_path)

    assert os.path.exists(opts.model_dir), 'Model not found.. %s does not exist!' % opts.model_dir

    assert isinstance(opts.batch_size, int)
    assert opts.batch_size > 0
    assert isinstance(opts.device, str)

    return opts


def ffwd(data_in, paths_out, model_dir, device_t='/gpu:0', batch_size=4):

    # read in img
    img = imread(data_in[0], mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    # get img_shape
    img_shape = img.shape

    # start TF graph
    g = tf.Graph()
    # define batch_size
    batch_size = min(len(paths_out), batch_size)
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    # TF session
    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:

        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        # get predictions from vgg model
        preds = transform.net(img_placeholder)
        saver = tf.train.Saver()

        # read pre-trained model
        if os.path.isdir(model_dir):
            ckpt = tf.train.get_model_state(model_dir)
            if ckpt and ckpt.model_model_path:
                saver.restore(sess, ckpt.model_model_path)
            else:
                raise Exception("No model found...")
        else:
            saver.restore(sess, model_dir)

        pos = 0
        curr_batch_out = paths_out[pos:pos + batch_size]
        # curr_batch_in = data_in[pos:pos + batch_size]
        X = np.zeros(batch_shape, dtype=np.float32)
        X[0] = img
        _preds = sess.run(preds, feed_dict={img_placeholder: X})
        for j, path_out in enumerate(curr_batch_out):
            img = np.clip(img, 0, 255).astype(np.uint8)
            imsave(path_out, img)


def ffwd_different_dimensions(in_path, out_path, model_dir, device, batch_size):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)

    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)

    for shape in in_path_of_shape:
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape],
             model_dir, device, batch_size)


def main():
    opts = get_opts()

    # check if input is file or dir
    if not os.path.isdir(opts.in_path):
        full_in = [opts.in_path]
        full_out = [os.path.join(opts.out_path, os.path.basename(opts.in_path)) if os.path.isdir(opts.out_path) \
                    else opts.out_path ]
    else:
        # get all filenames
        files = []
        for (dirpath, dirnames, filenames) in os.walk(opts.in_path):
            files.extend(filenames)

        full_in = [os.path.join(opts.in_path, x) for x in files]
        full_out = [os.path.join(opts.out_path, x) if os.path.isdir(opts.out_path) else opts.out_path for x in files]

    ffwd(full_in, full_out, opts.model_dir, device_t=opts.device, batch_size=opts.batch_size)


if __name__ == '__main__':
    main()
