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
                        dest='model_dir', help='dir or .ckpt file to load model from',
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
    parser.add_argument('-b','--batch-size', type=int,
                        dest='batch_size', help='batch size for feed-forwarding',
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

# read image using scipy
def read_img(src):
   img = imread(src, mode='RGB')
   if not (len(img.shape) == 3 and img.shape[2] == 3):
       img = np.dstack((img,img,img))

   return img

def eval(data_in, paths_out, model_dir, device='/gpu:0', batch_size=4):
    '''
    Transfers image style to another image using feed-forwarding and a pre-trained model

    :param data_in: List of input content images (having same shape)
    :param paths_out: List of output paths
    :param model_dir: Dir for input pre-trained model
    :param device: GPU to use for computation
    :param batch_size: Number of images batched (def: 4) or # of images if smaller
    :return: Stylized image(s)

    '''

    # read in img
    img = read_img(data_in[0])

    # get img_shape
    img_shape = img.shape

    # start TF graph
    g = tf.Graph()

    # define batch_size
    batch_size = min(len(paths_out), batch_size)
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    # TF session
    with g.as_default(), g.device(device), tf.Session(config=soft_config) as sess:

        batch_shape = (batch_size,) + img_shape
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

        # get predictions
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

        # iterate over batches (maybe run in parallel w joblib if needed)
        for i in range(num_iters):

            pos = i * batch_size
            curr_batch_out = paths_out[pos:pos + batch_size]
            curr_batch_in = data_in[pos:pos + batch_size]
            X = np.zeros(batch_shape, dtype=np.float32)

            # iterate over images in batch
            for j, path_in in enumerate(curr_batch_in):
                X[j] = read_img(path_in)
                _preds = sess.run(preds, feed_dict={img_placeholder: X})

            # save output images
            for j, path_out in enumerate(curr_batch_out):
                img = np.clip(_preds[j], 0, 255).astype(np.uint8)
                imsave(path_out, img)

        remaining_in = data_in[num_iters * batch_size:]
        remaining_out = paths_out[num_iters * batch_size:]

    # re-run on remaining images in list not in previous batch
    if len(remaining_in) > 0:
        eval(remaining_in, remaining_out, model_dir,
             device=device, batch_size=1)

def eval_mul_dims(in_path, out_path, model_dir, device, batch_size):
    '''
    Runs "eval" on diff image shapes after grouping them by shape
    '''
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)

    # if imgs have diff shapes, get all shapes
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%dx%d" % imread(in_image, mode='RGB').shape

        # group images by shape in dict
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)

    for shape in in_path_of_shape:

        # run function on every unique image shape
        eval(in_path_of_shape[shape], out_path_of_shape[shape],
             model_dir, device, batch_size)


def main():
    opts = get_opts()

    # check if input is file or dir
    if not os.path.isdir(opts.in_path):
        full_in = [opts.in_path]
        full_out = [os.path.join(opts.out_path, os.path.basename(opts.in_path)) if os.path.isdir(opts.out_path) \
                    else opts.out_path ]
    else:
        # get all filenames if dir
        files = []
        for (dirpath, dirnames, filenames) in os.walk(opts.in_path):
            files.extend(filenames)

        full_in = [os.path.join(opts.in_path, x) for x in files]
        full_out = [os.path.join(opts.out_path, x) if os.path.isdir(opts.out_path) else opts.out_path for x in files]

    eval_mul_dims(full_in, full_out, opts.model_dir, device=opts.device, batch_size=opts.batch_size)


if __name__ == '__main__':
    main()
