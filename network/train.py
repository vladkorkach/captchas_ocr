import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf

import network.core_trainer as core_trainer
import network.utils as utils

FLAGS = utils.FLAGS

logger = logging.getLogger('Training for captchas OCR using CNN+GRU+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=utils.FLAGS.train_dir, val_dir=utils.FLAGS.val_dir, mode='train'):
    """
    Train model functionality
    """
    model = core_trainer.OCRNetwork(mode)
    model.build_graph()

    train_feeder = utils.DataIterator(data_dir=train_dir)
    val_feeder = utils.DataIterator(data_dir=val_dir)

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)
    shuffle_idx_val = np.random.permutation(num_val_samples)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    # define data (weights, biases) distributions to plot with tensorboard
    names_to_plot = [
        "cnn/unit-1/cnn-1/W:0",
        "cnn/unit-1/cnn-1/b:0",
        "cnn/unit-2/cnn-2/W:0",
        "cnn/unit-2/cnn-2/b:0",
        "cnn/unit-3/cnn-3/W:0",
        "cnn/unit-3/cnn-3/b:0",
        "cnn/unit-4/cnn-4/W:0",
        "cnn/unit-4/cnn-4/b:0",
        "lstm/W_out:0",
        "lstm/b_out:0"
    ]

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        for name in names_to_plot:
            tf.summary.histogram(name, sess.graph.get_tensor_by_name(name))

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
        write_op = tf.summary.merge_all()

        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                # the global_step will restore sa well
                saver.restore(sess, ckpt)
                print('restore from checkpoint{0}'.format(ckpt))

        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx = np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()

            # the training part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch + 1) % 100 == 0:
                    print('batch', cur_batch, ': time', time.time() - batch_time)
                batch_time = time.time()
                indexs = [shuffle_idx[i % num_train_samples] for i in
                          range(cur_batch * FLAGS.batch_size, (cur_batch + 1) * FLAGS.batch_size)]
                batch_inputs, _, batch_labels = \
                    train_feeder.input_index_generate_batch(indexs)

                feed = {model.inputs: batch_inputs,
                        model.labels: batch_labels}

                # get summary for tensorboard visualization
                summary_str, batch_cost, step, _ = \
                    sess.run([model.merged_summary, model.cost, model.global_step, model.train_op], feed)
                # calculate the cost of operation for avg_loss metric
                train_cost += batch_cost * FLAGS.batch_size

                train_writer.add_summary(summary_str, step)

                summary = sess.run(write_op, feed_dict=feed)
                train_writer.add_summary(summary, step)
                train_writer.flush()

                # save the checkpoint
                if step % FLAGS.save_steps == 1:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save checkpoint at step {0}', format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'), global_step=step)
                    print('save checkpoint at step {0}'.format(step))

                if step % FLAGS.validation_steps == 0:
                    acc_batch_total = 0
                    dummy_acc_batch_total = 0
                    lastbatch_err = 0
                    lr = 0
                    for j in range(num_batches_per_epoch_val):
                        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in
                                      range(j * FLAGS.batch_size, (j + 1) * FLAGS.batch_size)]
                        val_inputs, _, val_labels = \
                            val_feeder.input_index_generate_batch(indexs_val)
                        val_feed = {model.inputs: val_inputs,
                                    model.labels: val_labels}

                        dense_decoded, lastbatch_err, lr, lgsts = \
                            sess.run([model.dense_decoded, model.cost, model.lrn_rate, model.logits],
                                     val_feed)

                        original_labels = val_feeder.the_label(indexs_val)
                        acc, dummy_acc = utils.accuracy_calculation(original_labels, dense_decoded,
                                                         ignore_value=-1, isPrint=True, epoch=cur_epoch + 1)
                        acc_batch_total += acc
                        dummy_acc_batch_total += dummy_acc

                    accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples
                    d_accuracy = (dummy_acc_batch_total * FLAGS.batch_size) / num_val_samples

                    summt = tf.Summary()
                    summt.value.add(tag="accuracy", simple_value=accuracy)
                    summt.value.add(tag="single letters accuracy", simple_value=d_accuracy)

                    train_writer.add_summary(summt, step)
                    train_writer.flush()

                    avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                          "accuracy = {:.4f},avg_train_cost = {:.8f}, " \
                          "lastbatch_err = {:.8f}, time = {:.3f},lr={:.8f}, letters_accuracy={:.8f}"
                    print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                     cur_epoch + 1, FLAGS.num_epochs, accuracy, avg_train_cost,
                                     lastbatch_err, time.time() - start_time, lr, d_accuracy))


def train_dispather():
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)
