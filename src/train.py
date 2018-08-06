from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ops
import data
import utils
import models
import argparse
import numpy as np
import tensorflow as tf
import image_utils as im

from datetime import datetime
from dateutil import tz

from glob import glob

import os

def load_data(args):
    img_ext = '/*.png'

    print("Using input directories:")

    if args.triplet:
        dir_train_phase1 = './datasets/' + args.dataset + '/train' + args.stage1 + img_ext
        dir_test_phase1 = './datasets/' + args.dataset + '/test' + args.stage1 + img_ext
        print(dir_train_phase1)
        print(dir_test_phase1)

        a_img_paths = glob(dir_train_phase1)
        a_data_pool = data.ImageData(sess, a_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)
        a_test_img_paths = glob(dir_test_phase1)
        a_test_pool = data.ImageData(sess, a_test_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)
    else:
        a_data_pool = a_test_pool = None
        
    dir_train_phase2 = './datasets/' + args.dataset + '/train' + args.stage2 + img_ext
    dir_train_phase3 = './datasets/' + args.dataset + '/train' + args.stage3 + img_ext
    dir_test_phase2 = './datasets/' + args.dataset + '/test' + args.stage2 + img_ext
    dir_test_phase3 = './datasets/' + args.dataset + '/test' + args.stage3 + img_ext

    print(dir_train_phase2)
    print(dir_train_phase3)
    print(dir_test_phase2)
    print(dir_test_phase3)

    b_img_paths = glob(dir_train_phase2)
    c_img_paths = glob(dir_train_phase3)

    b_data_pool = data.ImageData(sess, b_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)
    c_data_pool = data.ImageData(sess, c_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)

    b_test_img_paths = glob(dir_test_phase2)
    c_test_img_paths = glob(dir_test_phase3)
    b_test_pool = data.ImageData(sess, b_test_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)
    c_test_pool = data.ImageData(sess, c_test_img_paths, args.batch_size, load_size=args.load_size, crop_size=args.crop_size)

    return a_data_pool, b_data_pool, c_data_pool, a_test_pool, b_test_pool, c_test_pool

def save_single_img(a_real_ipt, b_real_ipt, save_dir, fname, forward_mapping=True):
    [a2b_opt] = sess.run([a2b if forward_mapping else b2a], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
    sample_opt = np.array(a2b_opt)
    utils.mkdir(save_dir + '/')
    targetDir = '%s/%s' % (save_dir, fname)
    im.imwrite(im.immerge(sample_opt,1,1), targetDir)

def training_run_id():
    datetime_s = datetime.now(tz.gettz('Europe/Helsinki')).strftime(r"%y%m%d_%H%M")
    return ""+datetime_s


def build_networks():
    with tf.device('/gpu:%d' % args.gpu_id):
        # Nodes
        a_real = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        b_real = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        a2b_sample = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        b2a_sample = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])

        a2b1 = models.generator(a_real, 'a2b')
        b2a1 = models.generator(b_real, 'b2a')

        if args.transform_twice: #a-b-c
            a2b = models.generator(a2b1, 'a2b', reuse=True)
            b2a = models.generator(b2a1, 'b2a', reuse=True)
        else:
            a2b = a2b1
            b2a = b2a1

        b2a2b = models.generator(b2a, 'a2b', reuse=True)
        a2b2a = models.generator(a2b, 'b2a', reuse=True)
        
        if args.transform_twice: #a-b-c
            b2a2b = models.generator(b2a2b, 'a2b', reuse=True)
            a2b2a = models.generator(a2b2a, 'b2a', reuse=True)

        # Add extra loss term to enforce the discriminator's power to discern A samples from B samples

        a_dis = models.discriminator(a_real, 'a')
        a_from_b_dis = models.discriminator(b_real, 'a', reuse=True) #mod1

        b2a_dis = models.discriminator(b2a, 'a', reuse=True)
        b2a_sample_dis = models.discriminator(b2a_sample, 'a', reuse=True)
        b_dis = models.discriminator(b_real, 'b')
        b_from_a_dis = models.discriminator(a_real, 'b', reuse=True) #mod1

        a2b_dis = models.discriminator(a2b, 'b', reuse=True)
        a2b_sample_dis = models.discriminator(a2b_sample, 'b', reuse=True)

        double_cycle_loss = 0.0

        if args.double_cycle: #Now making these double-processed samples belong to the same domain as 1-processed. I.e. the domains are "reflexive".
            a2b_sample_dis2 = models.discriminator(models.generator(a2b_sample, 'a2b', reuse=True), 'b', reuse=True)
            b2a_sample_dis2 = models.discriminator(models.generator(b2a_sample, 'b2a', reuse=True), 'a', reuse=True)

            a2b2b = models.generator(a2b, 'a2b', reuse=True)
            a2b2b2a = models.generator(a2b2b, 'b2a', reuse=True)
            a2b2b2a2a = models.generator(a2b2b2a, 'b2a', reuse=True)
            b2a2a = models.generator(b2a, 'b2a', reuse=True)
            b2a2a2b = models.generator(b2a2a, 'a2b', reuse=True)
            b2a2a2b2b = models.generator(b2a2a2b, 'a2b', reuse=True)

            cyc_loss_a2 = tf.identity(ops.l1_loss(a_real, a2b2b2a2a) * 10.0, name='cyc_loss_a2')
            cyc_loss_b2 = tf.identity(ops.l1_loss(b_real, b2a2a2b2b) * 10.0, name='cyc_loss_b2')

            double_cycle_loss = cyc_loss_a2 + cyc_loss_b2

        # Losses
        g_loss_a2b = tf.identity(ops.l2_loss(a2b_dis, tf.ones_like(a2b_dis)), name='g_loss_a2b')
        g_loss_b2a = tf.identity(ops.l2_loss(b2a_dis, tf.ones_like(b2a_dis)), name='g_loss_b2a')
        cyc_loss_a = tf.identity(ops.l1_loss(a_real, a2b2a) * 10.0, name='cyc_loss_a')
        cyc_loss_b = tf.identity(ops.l1_loss(b_real, b2a2b) * 10.0, name='cyc_loss_b')
        g_loss = g_loss_a2b + g_loss_b2a + cyc_loss_a + cyc_loss_b + double_cycle_loss

        d_loss_b2a_sample2 = d_loss_a2b_sample2 = 0.0

        d_loss_a_real = ops.l2_loss(a_dis, tf.ones_like(a_dis))
        d_loss_a_from_b_real = tf.identity(ops.l2_loss(a_from_b_dis, tf.zeros_like(a_from_b_dis)), name='d_loss_a_from_b') #mod1

        d_loss_b2a_sample = ops.l2_loss(b2a_sample_dis, tf.zeros_like(b2a_sample_dis))
        if args.double_cycle:
            d_loss_b2a_sample2 = ops.l2_loss(b2a_sample_dis2, tf.zeros_like(b2a_sample_dis))

        d_loss_a = tf.identity((d_loss_a_real + d_loss_b2a_sample + d_loss_b2a_sample2 + d_loss_a_from_b_real) / 3.0, name='d_loss_a')
        d_loss_b_real = ops.l2_loss(b_dis, tf.ones_like(b_dis))
        d_loss_b_from_a_real = tf.identity(ops.l2_loss(b_from_a_dis, tf.zeros_like(b_from_a_dis)), name='d_loss_b_from_a') #mod1

        d_loss_a2b_sample = ops.l2_loss(a2b_sample_dis, tf.zeros_like(a2b_sample_dis))
        if args.double_cycle:
            d_loss_a2b_sample2 = ops.l2_loss(a2b_sample_dis2, tf.zeros_like(a2b_sample_dis))  

        d_loss_b = tf.identity((d_loss_b_real + d_loss_a2b_sample + d_loss_a2b_sample2 + d_loss_b_from_a_real) / 3.0, name='d_loss_b')

        # Summaries
        g_summary   = ops.summary_tensors([g_loss_a2b, g_loss_b2a, cyc_loss_a, cyc_loss_b])
        d_summary_a = ops.summary_tensors([d_loss_a, d_loss_a_from_b_real])
        d_summary_b = ops.summary_tensors([d_loss_b, d_loss_b_from_a_real])

        # Optim
        t_var = tf.trainable_variables()
        d_a_var = [var for var in t_var if 'a_discriminator' in var.name]
        d_b_var = [var for var in t_var if 'b_discriminator' in var.name]
        g_var   = [var for var in t_var if 'a2b_generator'   in var.name or 'b2a_generator' in var.name]

        d_a_train_op = tf.train.AdamOptimizer(args.lr, beta1=0.5).minimize(d_loss_a, var_list=d_a_var)
        d_b_train_op = tf.train.AdamOptimizer(args.lr, beta1=0.5).minimize(d_loss_b, var_list=d_b_var)
        g_train_op = tf.train.AdamOptimizer(args.lr, beta1=0.5).minimize(g_loss, var_list=g_var)

        return g_train_op, d_a_train_op, d_b_train_op, g_summary, d_summary_a, d_summary_b, a2b, a2b2a, b2a, b2a2b, a_real, b_real, a2b_sample, b2a_sample, a2b1, b2a1

def get_args():  
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', dest='dataset', required=True, help='Dataset directory that contains the trainABC, trainDEF, trainABC and testDEF directories, where ABC and DEF stand for the stage1 and stage2 arguments')
    parser.add_argument('--checkpointroot', dest='checkpointroot', default='./checkpoints', help='Directory for storing checkpoints')
    parser.add_argument('--prev_checkpoint', dest='prev_checkpoint', default=None, help='Use the specific checkpoint of the form "Epoch_(256)_(1828of2337)_step_(600099)" (no ckpt).')

    parser.add_argument('--load_size', dest='load_size', type=int, default=256, help='scale images to this size')
    parser.add_argument('--crop_size', dest='crop_size', type=int, default=256, help='then crop to this size')
    parser.add_argument('--max_epochs', dest='epoch', type=int, default=200, help='# of epochs to run')
    parser.add_argument('--max_steps', dest='max_steps', type=int, default=1e9, help='# of max training steps to take')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in a batch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--gpu_id', dest='gpu_id', type=int, default=0, help='GPU ID')

    parser.add_argument('--stage1', dest='stage1', required=True, help='[dataset]/[train|test][stage-name] for stage #1, e.g. "_age_35" if your dataset is "celeb" and data is under celeb/train_age35')
    parser.add_argument('--stage2', dest='stage2', required=True, help='[dataset]/[train|test][stage-name] for stage #2')
    parser.add_argument('--stage3', dest='stage3', default=None, help='[dataset]/[train|test][stage-name] for stage #3')
    parser.add_argument('--subnet', dest='subnet', default='', help='Sub-network to use for this transformer, with separate checkpoints')

    # The special modes:
    parser.add_argument('--double_cycle', dest='double_cycle', action='store_true', default='', help='Constraint to ensure that if you run a transformer twice in succession, and then twice in reverse, you get back the original.')
    parser.add_argument('--triplet', dest='triplet', default='', action='store_true', help='Run the transitive transformation on both the previous and current dataset.')
    parser.add_argument('--transform_twice', dest='transform_twice', action='store_true', default='', help='Run the transitive transformation twice on the source data set. Maintain consistency.')

    # Tests:
    parser.add_argument('--chaintestdir', dest='chaintestdir', default=None, help='Show single and double transformations for images in the given directory. Use the latest triplet weights (must exist).')
    parser.add_argument('--singletestN', dest='singletestN', type=int, default=0, help='Show the given number of single transformations.')
    parser.add_argument('--singletestdir', dest='singletestdir', default="", help='Input dir for the single transformation test.')
    parser.add_argument('--singletestdir_out', dest='singletestdir_out', default="", help='Output dir for the single transformation test.')
    parser.add_argument('--singletest_forward', dest='singletest_forward', type=int, default=1, help='Map the images to forward/backward direction (0 = backward, 1 = forward)')

    parser.add_argument('--samplingcycle', dest='samplingcycle', type=int, default=0, help='How often to generate a transformed sample batch from TEST set, e.g. for age estimation.')
    parser.add_argument('--online_sampling_batch_size', dest='online_sampling_batch_size', type=int, default=50, help="Number of samples for the auxiliary samples")

    parser.add_argument('--save', dest='do_save', type=int, required=True, help='Save weights after training (0/1)')

    args = parser.parse_args()

    assert((not args.triplet) or args.stage3)

    # In terms of actual variable naming, we assume the triplet mode is the standard - so that if we do not use triplets, we start from stage #2
    if not args.stage3:
        args.stage3 = args.stage2
        args.stage2 = args.stage1

    assert(args.stage2 != args.stage3)

    return args

# Input args handling

args = get_args()

print(args)

do_train = (args.chaintestdir == None and args.singletestN <= 0)
print("Will train: " + str(do_train))

do_save = (args.do_save==1)
print("Will save weights after training: " + str(do_save))

singleTestOnly = len(args.singletestdir) > 0

if args.triplet:
    print("Triplet enabled. You intend to apply a network trained on {}->{} on data {} while maintaining compatibility with the previous transform.".format(args.stage1, args.stage2, args.stage3))

# Network building

g_train_op, d_a_train_op, d_b_train_op, g_summary, d_summary_a, d_summary_b, a2b, a2b2a, b2a, b2a2b, a_real, b_real, a2b_sample, b2a_sample, a2b1, b2a1 = build_networks()

# Session management

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
it_cnt, update_cnt = ops.counter()

if do_train:
    summary_writer = tf.summary.FileWriter('./summaries/' + args.dataset+ '/train-'+training_run_id(), sess.graph)

# Data loading

if not singleTestOnly:
    a_data_pool, b_data_pool, c_data_pool, a_test_pool, b_test_pool, c_test_pool = load_data(args)
else:
    single_test_input_pool = data.ImageData(sess, glob(args.singletestdir+'/*.png'), 1, load_size=args.load_size, crop_size=args.crop_size, shuffle = False, random_flip = True) #Fix the random flip problem, see data.py, then make the flip False.

b2c_pool = utils.ItemPool()
c2b_pool = utils.ItemPool()
a2b_pool = utils.ItemPool()
b2a_pool = utils.ItemPool()

# Checkpoint management.

saver = tf.train.Saver(max_to_keep=5)

# If the triplet mode is enabled, we try to load the existing checkpoint for that first.
# Otherwise, we try to load the regular checkpoint only.

subnet_maybe        = ('/'+args.subnet) if len(args.subnet) > 0 else ''
subnet_ext_maybe    = (subnet_maybe + ('-transitive2')) if args.transform_twice else subnet_maybe
ckpt_dir_normal     = args.checkpointroot + '/' + args.dataset + subnet_maybe
ckpt_dir_ext        = args.checkpointroot + '/' + args.dataset + subnet_ext_maybe
online_samples_dir  = './sample_images_while_training/' + args.dataset + subnet_ext_maybe
utils.mkdir(online_samples_dir + '/')

ckpt_path = None
ckpt_dir = None

#TODO: The prev_checkpoint does not support the transform_twice directories yet
if (not do_train or not do_save) and args.prev_checkpoint:
    prev_ckpt = args.prev_checkpoint+".ckpt"
    print("Loading precise checkpoint {}/{}".format(ckpt_dir_normal, prev_ckpt))
    saver.restore(sess, os.path.join(ckpt_dir_normal, prev_ckpt))
else:
    utils.mkdir(ckpt_dir_normal + '/')
    if args.transform_twice:
        print("Transform-twice mode weight loading attempting...")
        utils.mkdir(ckpt_dir_ext + '/')
        ckpt_dir = ckpt_dir_ext
        ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
    else:
        print("No Transform-twice mode weight loading attempted.")

    if ckpt_path is None:
        print("No Transform-twice mode weight loading done. Attempting regular weight loading...")
        ckpt_dir = ckpt_dir_normal
        ckpt_path = utils.load_checkpoint(ckpt_dir, sess, saver)
        if ckpt_path is None:
            print("No checkpoints found for loading.")
            if args.transform_twice or args.triplet:
                print("You requested a re-use mode but there were no existing weights of a subnet. Did you specify an existing subnet?")
                sys.exit()
            sess.run(tf.global_variables_initializer())
            print("In the future, using checkpoint directory {}".format(ckpt_dir_normal))
        else:
            print('Loading checkpoint from Copy variables from % s' % ckpt_path)
            if args.transform_twice:
                ckpt_dir = ckpt_dir_ext
                print("Saving checkpoints of this session under {}".format(ckpt_dir))

# Train / test

try:
    tf_coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=tf_coord)

    epoch = args.epoch

    if do_train:
        if not args.triplet:
            batch_epoch = len(b_data_pool) // args.batch_size
        else:
            batch_epoch = min(len(a_data_pool), len(b_data_pool)) // args.batch_size
        max_it = min(epoch * batch_epoch, args.max_steps)
        last_it = -1
        for it in range(sess.run(it_cnt), max_it):
            sess.run(update_cnt)

            last_it = it

            # prepare data            
            b_real_ipt_orig = b_data_pool.batch()
            c_real_ipt_orig = c_data_pool.batch()

            matching_domain_sample_pairs = [(b_real_ipt_orig, c_real_ipt_orig)]
            if args.triplet:
                a_real_ipt_orig = a_data_pool.batch()
                matching_domain_sample_pairs += [(a_real_ipt_orig, b_real_ipt_orig)]

            for a_real_ipt, b_real_ipt in matching_domain_sample_pairs:
                a2b_opt, b2a_opt = sess.run([a2b, b2a], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
                a2b_sample_ipt = np.array(a2b_pool(list(a2b_opt)))
                b2a_sample_ipt = np.array(b2a_pool(list(b2a_opt)))

                # train G
                g_summary_opt, _   = sess.run([g_summary, g_train_op],     feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
                summary_writer.add_summary(g_summary_opt, it)
                # train D_b
                d_summary_b_opt, _ = sess.run([d_summary_b, d_b_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt, a2b_sample: a2b_sample_ipt})
                summary_writer.add_summary(d_summary_b_opt, it)
                # train D_a
                d_summary_a_opt, _ = sess.run([d_summary_a, d_a_train_op], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt, b2a_sample: b2a_sample_ipt})
                summary_writer.add_summary(d_summary_a_opt, it)

            epoch = it // batch_epoch
            it_epoch = it % batch_epoch + 1

            # display
            print("Epoch: (%3d) (%5d/%5d) %s" % (epoch, it_epoch, batch_epoch, '(a->b->c)' if args.triplet else ''))

            # Checkpointing
            if do_save and (it + 1) % 10000 == 0:
                save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d)_step_(%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch,it))
                print('Model saved in file: % s' % save_path)

            # Sample images for external evaluation (i.e. just raw single images). Note: For triplet=true, there are 2x steps involved.
            if args.samplingcycle > 0 and (it % args.samplingcycle == 0) and it > 0:
                print("Create samples for the external evaluator (aux batch {} with size {})".format(int(it/args.samplingcycle), args.online_sampling_batch_size))
                for c_i in range(args.online_sampling_batch_size):
                    fname = 'Transformed_from_%s_(%dof%d)_once.png' % (args.stage2, c_i, args.singletestN)
                    save_single_img(a_real_ipt = b_test_pool.batch(), b_real_ipt = c_test_pool.batch(), save_dir = './aux_samples/' + args.dataset + subnet_ext_maybe+'/'+args.stage2+'/'+str(int(it)), fname=fname)

            # Create sample images with a-b-a structure
            if (it + 1) % 100 == 0:
                a_real_ipt = b_test_pool.batch()
                b_real_ipt = c_test_pool.batch()
                [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
                sample_opt = np.concatenate((a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt, b2a_opt, b2a2b_opt), axis=0)               

                im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d).png' % (online_samples_dir, epoch, it_epoch, batch_epoch))

                if args.double_cycle:
                    [a2b_opt, a2b2a_opt, b2a_opt, b2a2b_opt] = sess.run([a2b, a2b2a, b2a, b2a2b], feed_dict={a_real: a2b_opt, b_real: b2a_opt})
                    sample_opt = np.concatenate((a_real_ipt, a2b_opt, a2b2a_opt, b_real_ipt, b2a_opt, b2a2b_opt), axis=0)
                    im.imwrite(im.immerge(sample_opt, 2, 3), '%s/Epoch_(%d)_(%dof%d)_double_cycle.png' % (online_samples_dir, epoch, it_epoch, batch_epoch))

        if do_save and last_it != -1:
            save_path = saver.save(sess, '%s/Epoch_(%d)_(%dof%d)_step_(%d).ckpt' % (ckpt_dir, epoch, it_epoch, batch_epoch,last_it))
            print('Final model saved in file: % s' % save_path)

    elif args.chaintestdir:
        chaintests_N = 20
        print("Run chain test on dir {} for {} times".format(args.chaintestdir, chaintests_N))
        for c_i in range(chaintests_N):
            a_real_ipt = b_test_pool.batch()
            b_real_ipt = c_test_pool.batch()
            [a2b_opt, a2b1_opt] = sess.run([a2b, a2b1], feed_dict={a_real: a_real_ipt, b_real: b_real_ipt})
            sample_opt = np.concatenate((a_real_ipt, a2b1_opt, a2b_opt), axis=0)
            im.imwrite(im.immerge(sample_opt, 1, 3), '%s/Epoch_(%d)_(%dof%d)_once_and_twice.png' % (online_samples_dir, epoch, c_i, chaintests_N))

    elif singleTestOnly:
        print("Run single imgs test for {} times in direction {}".format(args.singletestN, "FORWARD" if args.singletest_forward==1 else "REVERSE"))
        for c_i in range(args.singletestN):
            fname = 'Transformed_from_%s_(%dof%d)_once.png' % (args.stage2, c_i, args.singletestN)            
            test_batch = single_test_input_pool.batch()            
            _save_dir = args.singletestdir_out if (not args.singletestdir_out == None) else online_samples_dir + '/s'

            save_single_img(a_real_ipt = test_batch, b_real_ipt = test_batch, save_dir = _save_dir, fname=fname, forward_mapping = (args.singletest_forward == 1))

except Exception, e:
    tf_coord.request_stop(e)
    print(e)
finally:
    print("Stop threads and close session!")
    tf_coord.request_stop()
    tf_coord.join(threads)
    sess.close()
