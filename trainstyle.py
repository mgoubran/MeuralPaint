from __future__ import print_function
import os
import numpy as np
from scipy.misc import imread, imsave
from optimize import optimize
from argparse import ArgumentParser
import quickpaint
import glob

DEVICE = '/gpu:0'
FRAC_GPU = 1

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-dir', type=str, default='checkpoint',
                        dest='checkpoint_dir', help='dir to save checkpoint in',
                        metavar='CHECKPOINT_DIR', required=True)
    parser.add_argument('-s', '--style', type=str,
                        dest='style', help='desired style image path',
                        metavar='STYLE', required=True)
    parser.add_argument('-t', '--train-path', type=str, default='data/train2014',
                        dest='train_path', help='path to training images folder',
                        metavar='TRAIN_PATH')
    parser.add_argument('-o', '--output', type=str,
                        dest='output', help='output test image at every checkpoint path',
                        metavar='OUTPUT', default=False)
    parser.add_argument('-od', '--output-dir', type=str,
                        dest='output_dir', help='output test images dir',
                        metavar='OUTPUT DIR', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=2,
                        dest='epochs', help='# of epochs', metavar='EPOCHS')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        dest='batch_size', help='batch size',
                        metavar='BATCH_SIZE')
    parser.add_argument('-i', '--checkpoint-iterations', type=int, default=2000,
                        dest='checkpoint_iterations', help='checkpoint frequency',
                        metavar='CHECKPOINT_ITERATIONS')
    parser.add_argument('-n', '--net-path', type=str, default='data/imagenet-vgg-verydeep-19.mat',
                        dest='net_path', help='path to VGG19 network (default %(default)s)',
                        metavar='NET_PATH')
    parser.add_argument('-cw', '--content-weight', type=float, default=7.5e0,
                        dest='content_weight', help='content weight (default %(default)s)',
                        metavar='CONTENT_WEIGHT')
    parser.add_argument('-sw', '--style-weight', type=float, default=1e2,
                        dest='style_weight', help='style weight (default %(default)s)',
                        metavar='STYLE_WEIGHT')
    parser.add_argument('-tw', '--tv-weight', type=float, default=2e2,
                        dest='tv_weight', help='total variation regularization weight (default %(default)s)',
                        metavar='TV_WEIGHT')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3,
                        dest='learning_rate', help='learning rate (default %(default)s)',
                        metavar='LEARNING_RATE')

    opts = parser.parse_args()

    # check opts        
    assert os.path.exists(opts.style), 'style image not found.. %s does not exist!' % opts.style
    assert os.path.exists(opts.train_path), 'train path not found.. %s does not exist!' % opts.train_path
    assert os.path.exists(opts.net_path), 'Network not found.. %s does not exist!' % opts.net_path
    assert os.path.exists(opts.output), 'Output test not found.. %s does not exist' % opts.output

    if not os.path.exists(opts.output_dir):
        print('creating output tests dir')
        os.makedirs(opts.output_dir)

    if os.path.isdir(opts.checkpoint_dir):
        if not os.path.exists(opts.checkpoint_dir):
            print('creating checkpoints dir')
            os.makedirs(opts.checkpoint_dir)

    assert opts.epochs > 0
    assert opts.batch_size > 0
    assert opts.checkpoint_iterations > 0
    assert opts.content_weight >= 0
    assert opts.style_weight >= 0
    assert opts.tv_weight >= 0
    assert opts.learning_rate >= 0

    return opts


# read image using scipy
def read_img(src):
    img = imread(src, mode='RGB')
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    return img

def main():
    opts = get_opts()

    style_target = read_img(opts.style)
    content_targets = glob.glob('%s/*' % opts.train_path)
    style_name = os.path.split(os.path.basename(opts.style))[0]

    kwargs = {"epochs": opts.epochs, "print_iterations": opts.checkpoint_iterations,
              "batch_size": opts.batch_size, "save_path": os.path.join(opts.checkpoint_dir, '%s.ckpt' % style_name),
              "learning_rate": opts.learning_rate}

    args = [content_targets, style_target, opts.content_weight, opts.style_weight, opts.tv_weight,
            opts.net_path]

    for preds, losses, i, epoch in optimize(*args, **kwargs):
        style_loss, content_loss, tv_loss, loss = losses

        print('Epoch %d, Iteration: %d, Loss: %s' % (epoch, i, loss))
        to_print = (style_loss, content_loss, tv_loss)

        print('style: %s, content:%s, tv: %s' % to_print)
        if opts.output:
            preds_path = '%s/%s_%s.png' % (opts.output_dir, epoch, i)

            quickpaint.eval_mul_dims(opts.output, preds_path, opts.checkpoint_dir)

    cmd_text = 'python quickpaint.py --checkpoint %s ...' % opts.checkpoint_dir
    print("Training complete. For evaluation:\n    `%s`" % cmd_text)


if __name__ == '__main__':
    main()
