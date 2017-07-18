#!/usr/bin/env python

# coding: utf-8

import transform
import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
from collections import defaultdict
from scipy.misc import imread, imsave
import time
import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']="3"  # filter out info & warning logs

# read input arguments
def get_opts():
    parser = ArgumentParser(description="Paint (transfer style to) image using a pre-trained neural network model.",
                            formatter_class=RawTextHelpFormatter,
                            usage="./quickpaint.py -i [ input (content) ] -o [ output (stylized content) ] -m [ model "
                                  "(style) ] -ma [ mask ] -bl [ blend ]"
                                  "Example: ./quickpaint.py -i inputs/stanford.jpg -o outputs/stanford_cubist.jpg -m "
                                  "pre-trained_models/cubist.model -ma 1 -bl 0.5 ")
    parser.add_argument('-m', '--model', type=str,
                        dest='model_path', help='path to load model (.ckpt or .model/.meta)',
                        metavar='MODEL', required=True)
    parser.add_argument('-i', '--input', type=str,
                        dest='in_path', help='dir or file to transform (content)',
                        metavar='IN_PATH', required=True)
    parser.add_argument('-o', '--output', type=str,
                        dest='out_path', help='destination (dir or file) of transformed input (stylized content)',
                        metavar='OUT_PATH', required=True)
    parser.add_argument('-d', '--device', type=str,
                        dest='device', help='device to perform compute on (default: %(default)s)',
                        metavar='', default='/gpu:0')
    parser.add_argument('-b', '--batch-size', type=int,
                        dest='batch_size', help='batch size for feed-forwarding (default: %(default)s)',
                        metavar='', default=4)
    parser.add_argument('-a', '--model-arch', type=str,
                        dest='model_arch',
                        help='model architecture if models in form (.model) are used, (default: %(default)s)',
                        metavar='', default='pre-trained_models/model.meta')
    parser.add_argument('-ma', '--mask', type=int,
                        dest='mask', help='create binary mask from input (@ 1% of max) and mask output, (default: %(default)s)',
                        metavar='', default=0)
    parser.add_argument('-bl', '--blend', type=float,
                        dest='blend', help='multiply the original image with the output using a weighting (factor), (default: %(default)s)',
                        metavar='', default=0)

    opts = parser.parse_args()

    # check inputs
    assert os.path.exists(opts.in_path), 'Input dir: %s does not exist!' % opts.in_path

    if os.path.isdir(opts.out_path):
        if not os.path.exists(opts.out_path):
            print('creating output dir')
            os.makedirs(opts.out_path)

    assert os.path.exists(opts.model_path), 'Model not found.. %s does not exist!' % opts.model_path

    assert isinstance(opts.batch_size, int), '-b, --batch_size needs to be a positive integer'
    assert opts.batch_size > 0, '-b, --batch_size needs to be a positive integer'
    assert isinstance(opts.device, str)
    assert (opts.mask==1 or opts.mask==0), '-ma, --mask needs to be binary'
    assert opts.blend <= 1, '-bl, --blend needs to be a float equal or less to 1'

    return opts


# read image using scipy
def read_img(src):
    img = imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))

    return img


def eval(data_in, paths_out, model_path, device='/gpu:0', batch_size=4,
         model_arch='pre-trained_models/model.meta', mask=0, blend=0):
    '''
    Transfers image style to another image using feed-forwarding and a pre-trained model

    :param data_in: List of input content images (having same shape)
    :param paths_out: List of output paths
    :param model_path: Path for input pre-trained model (either .ckpt or .model)
        for .model models will read model meta graph from pre-trained_modes/model.meta
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

        model_ext = os.path.splitext(model_path)[1]

        if model_ext == ".ckpt":

            img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

            # get predictions from model
            preds = transform.net(img_placeholder)
            saver = tf.train.Saver()
            # restore model
            saver.restore(sess, model_path)

        else:

            # print("\n Reading model architecture from %s" % opts.model_arch)

            # load model meta graph
            saver = tf.train.import_meta_graph(model_arch, clear_devices=True)
            # restore model
            saver.restore(sess, model_path)

            img_placeholder = tf.get_collection("inputs")[0]
            preds = tf.get_collection("output")[0]

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
                img = np.clip(_preds[j], 0, 255).astype(np.uint8) # after clipping to 255

                if mask == 1:
                    thr = X[i].max() * 0.01
                    inmask = np.where(X[j]>thr, 1, 0)
                    if inmask.shape != img.shape:
                        img = img[0:inmask.shape[0],0:inmask.shape[1],:]
                    img = np.multiply(inmask,img)                        
                
                if blend > 0:
                    inimg = X[i] * blend
                    if inimg.shape != img.shape:
                        img = img[0:inimg.shape[0],0:inimg.shape[1],:]
                    img = np.multiply(inimg,img)

                imsave(path_out, img)

        remaining_in = data_in[num_iters * batch_size:]
        remaining_out = paths_out[num_iters * batch_size:]

    # re-run on remaining images in list not in previous batch
    if len(remaining_in) > 0:
        eval(remaining_in, remaining_out, model_path,
             device=device, batch_size=1)


def eval_mul_dims(in_path, out_path, model_path, device, batch_size, model_arch, mask, blend):
    '''
    Runs "eval" on diff image shapes after grouping them by shape
    '''
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)

    # if images have diff shapes, get all shapes
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
             model_path, device, batch_size, model_arch, mask, blend)


def main():
    opts = get_opts()

    start_time = time.time()

    # check if input is file or dir
    if not os.path.isdir(opts.in_path):
        full_in = [opts.in_path]
        in_name = os.path.splitext(os.path.basename(opts.in_path))
        out_name = str(in_name[0] + '_' + os.path.splitext(os.path.basename(opts.model_path))[0] + in_name[1])
        full_out = [os.path.join(opts.out_path, out_name) if os.path.isdir(opts.out_path) else opts.out_path]
    else:
        # get all file names if dir
        files = []
        for (dir_path, dir_names, file_names) in os.walk(opts.in_path):
            files.extend(file_names)

        full_in = [os.path.join(opts.in_path, x) for x in files]
        full_out = [os.path.join(opts.out_path, x) if os.path.isdir(opts.out_path) else opts.out_path for x in files]

    eval_mul_dims(full_in, full_out, opts.model_path, device=opts.device, batch_size=opts.batch_size,
                  model_arch=opts.model_arch, mask=opts.mask, blend=opts.blend)

    print("\n Painting done in %0.3f seconds ... Have a good day!\n" % (time.time() - start_time))


if __name__ == '__main__':
    main()
