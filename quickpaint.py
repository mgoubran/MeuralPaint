#!/usr/bin/env python
# Maged Goubran @ 2016, mgoubran@stanford.edu

# coding: utf-8

from __future__ import print_function
import transform
import numpy as np
import os
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import defaultdict

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

    # check args
    exists(opts.model_dir, 'Model not found!')
    exists(opts.in_path, 'Input path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'Output dir not found!')
        assert opts.batch_size > 0

    return opts


def ffwd(data_in, paths_out, model_dir, device_t='/gpu:0', batch_size=4):

    assert len(paths_out) > 0
    assert len(data_in) == len(paths_out)
    img_shape = get_img(data_in[0]).shape # get img_shape

    g = tf.Graph() # start tensorflow graph
    batch_size = min(len(paths_out), batch_size)
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess: # TF session

        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
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

        num_iters = int(len(paths_out) / batch_size)

        for i in range(num_iters):
            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos + batch_size]
            if is_paths:
                curr_batch_in = data_in[pos:pos + batch_size]
                X = np.zeros(batch_shape, dtype=np.float32)
                for j, path_in in enumerate(curr_batch_in):
                    img = get_img(path_in)
                    assert img.shape == img_shape, \
                        'Images have different dimensions. ' + \
                        'Resize images or use --allow-different-dimensions.'
                    X[j] = img
            else:
                X = data_in[pos:pos + batch_size]

            _preds = sess.run(preds, feed_dict={img_placeholder: X})
            for j, path_out in enumerate(curr_batch_out):
                save_img(path_out, _preds[j])

        remaining_in = data_in[num_iters * batch_size:]
        remaining_out = paths_out[num_iters * batch_size:]
    if len(remaining_in) > 0:
        ffwd(remaining_in, remaining_out, model_dir,
             device_t=device_t, batch_size=1)


def ffwd_to_img(in_path, out_path, model_dir, device):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, model_dir, batch_size=1, device_t=device)


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
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = \
                os.path.join(opts.out_path, os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path

        ffwd_to_img(opts.in_path, out_path, opts.model_dir,
                    device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path, x) for x in files]
        full_out = [os.path.join(opts.out_path, x) for x in files]
        # if opts.allow_different_dimensions:
        ffwd_different_dimensions(full_in, full_out, opts.model_dir,
                                  device=opts.device, batch_size=opts.batch_size)
        # else:
        #     ffwd(full_in, full_out, opts.model_dir, device_t=opts.device,
        #          batch_size=opts.batch_size)


if __name__ == '__main__':
    main()
