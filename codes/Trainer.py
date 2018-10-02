import tensorflow as tf
import numpy as np
import cv2

import time
import os
from progress.bar import Bar

import config

slim = tf.contrib.slim

def average_gradients(grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

class Trainer:
    def __init__(self):
        print("Initializing Trainer")

        with tf.device('/cpu:0'):
            self.all_gradients = []
            self.all_losses = []
            self.all_outputs = []
            self.all_towers = []
            self.softmaxs = []
            self.network = config.NETWORK

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.7)#tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=0.1)#

            self.input = tf.placeholder(
                dtype=tf.float32,
                shape=[ None, config.IMAGE_FRAMES, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL],
                name='input'
            )

            self.label = tf.placeholder(dtype=tf.float32, shape=[None, config.NUM_OF_CLASS], name='label')

            self.tower_inputs = tf.split(self.input, config.NUM_GPUS, 0)
            self.tower_labels = tf.split(self.label, config.NUM_GPUS, 0)

            for i in range(config.NUM_GPUS):
                self.build_tower(i, self.tower_inputs[i], self.tower_labels[i])

            self.train_step = tf.group(
                self.optimizer.apply_gradients(
                    global_step = self.global_step,
                    grads_and_vars = average_gradients(self.all_gradients)
                )
            )

            self.global_loss = tf.reduce_mean(self.all_losses)

            self.softmax_val = tf.concat(self.softmaxs, axis=0)
            self.is_output = tf.argmax(tf.concat(self.all_outputs, axis=0), 1)
            self.is_ground = tf.argmax(tf.concat(self.tower_labels, axis=0), 1)
            self.is_correct = tf.equal(self.is_output, self.is_ground)
            self.global_acc = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

            self.summary = self.define_summarizer()
            print ("Initializing Done")

    def define_summarizer(self):
        """
        Define summary information for tensorboard
        """
        tf.summary.scalar('learning_rate', self.learning_rate)
        tf.summary.scalar('global_loss', self.global_loss)
        #tf.summary.image('filter', self.all_outputs)
        #tf.summary.scalar('global_mse', self.global_mse)


        return tf.summary.merge_all()

    def build_tower(self, gpu_index, X, Y):
        print('[!] BUILD TOWER %d' % gpu_index)
        with tf.device('/gpu:%d' % gpu_index), tf.name_scope('tower_%d' % gpu_index), tf.variable_scope(
                tf.get_variable_scope(), reuse=gpu_index is not 0):
            graph = config.MODEL_NETWORK.build_graph(X, Y, gpu_index is not 0, self.network)
            loss = graph['loss']
            output = graph['prediction']
            softmax = graph['softmax']
            gradients = self.optimizer.compute_gradients(loss)

            self.all_towers.append(graph)
            self.all_losses.append(loss)
            self.all_outputs.append(output)
            self.all_gradients.append(gradients)
            self.softmaxs.append(softmax)
            tf.get_variable_scope().reuse_variables()

    def get_learning_rate(self, step_index=0):
        init_learning_rate = float(config.TRAIN_LEARNING_RATE)
        decay = float(config.TRAIN_DECAY)
        decay_level = int(step_index/config.TRAIN_DECAY_INTERVAL)

        return init_learning_rate * (decay ** decay_level)

    def export_results(self, session, data_manager, is_train, sample_save_path):
        test_imgs = []
        test_labels = []
        pred_labels = []
        out_data = []
        softmax_val = []
        tot_acc = []
        # training evaluation

        # 7class 574, 125 batch 8; 1148, 250 batch 4
        # 3class 375, 128
        loop = config.EPOCH if is_train == True\
            else int((config.FILE_READ_LIMIT * config.NUM_OF_CLASS / config.BATCH_SIZE) - config.EPOCH)

        # get prediction results
        for i in range(loop):
            input = session.run([data_manager])
            X = np.reshape(input[0][0], [-1, config.IMAGE_FRAMES, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                         config.IMAGE_CHANNEL])  # 5D tensor (?, 10, 64, 64, 1)
            label = input[0][1]
            Y = []
            for j, value in enumerate(label):
                temp = [0] * config.NUM_OF_CLASS
                temp[value] = 1
                Y.append(temp)

            while (len(Y) < config.BATCH_SIZE):
                input = session.run([data_manager])
                X = np.concatenate(
                    (X, np.reshape(input[0][0], [-1, config.IMAGE_FRAMES, config.IMAGE_WIDTH, config.IMAGE_HEIGHT,
                                                 config.IMAGE_CHANNEL])))  # 5D tensor (?, 10, 64, 64, 1)
                label = input[0][1]
                for i, value in enumerate(label):
                    temp = [0] * config.NUM_OF_CLASS
                    temp[value] = 1
                    Y.append(temp)

            for j, x in enumerate(X):
                test_imgs.append(x.copy())
                test_labels.append(Y[j].copy())

            pred, acc, out, softmax = session.run(
                fetches=[self.is_correct,
                         self.global_acc,
                         self.is_output,
                         self.softmax_val],
                feed_dict={
                    self.input: X,
                    self.label: Y,
                    self.learning_rate: .0})
            pred_labels.append(pred)
            tot_acc.append(acc)
            out_data.append(out)
            for j in range(config.BATCH_SIZE):
                softmax_val.append([softmax[j][out[j]], softmax[j][label[j]]])

        acc_val = np.mean(tot_acc)

        if is_train:
            print('[Train ACC] ACC: %.4f' % acc_val)
        else:
            print('[Test ACC] ACC: %.4f' % acc_val)

        bar = Bar('Processing', max=loop * config.BATCH_SIZE, suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')

        false_save_path = sample_save_path+'/sample'
        false_save_path += '_train' if is_train == True else '_test'
        out_data = np.reshape(out_data, -1)
        softmax_val = np.reshape(softmax_val, (-1, 2))

        os.makedirs(false_save_path + '/original', exist_ok=True)
        for j in range(config.NUM_OF_CLASS):
            for k in range(config.NUM_OF_CLASS):
                dir_name = false_save_path + '/' + config.REV_MAP[j] + '/' + config.REV_MAP[k]
                os.makedirs(dir_name, exist_ok=True)

        fout = open(false_save_path + '/pred.txt', 'w')
        false_alarm = [0] * config.NUM_OF_CLASS ** 2
        false_alarm = np.reshape(false_alarm, (config.NUM_OF_CLASS, config.NUM_OF_CLASS))

        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        for index, video in enumerate(test_imgs):
            bar.next()
            k = np.argmax(test_labels[index])
            video_name = false_save_path + '/' + config.REV_MAP[k]\
                         + '/' + config.REV_MAP[out_data[index]]\
                         + '/' + str(softmax_val[index][0])\
                         + '_' + str(softmax_val[index][1])\
                         + '_' + str(index) + '.avi'
            writer = cv2.VideoWriter(video_name, fourcc, config.IMAGE_FRAMES, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            for frame in video:
                writer.write(frame)
            false_alarm[k][out_data[index]] += 1
            writer.release()
            video_name = false_save_path + '/original/' + str(index) + '.avi'
            writer = cv2.VideoWriter(video_name, fourcc, config.IMAGE_FRAMES, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
            for frame in video:
                writer.write(frame)
            writer.release()

        for i in range(config.NUM_OF_CLASS):
            fout.write('%s ' % config.REV_MAP[i])
            fout.write('\n')
            for j in range(config.NUM_OF_CLASS):
                fout.writelines('%d'% false_alarm[i][j])

        bar.finish()
        fout.close()

    def run_test(self, session):
        path = '/media/sda1/hyeon/vd_proj/database/global_30frames_2/tfrecord/test'
        test_vids = []
        test_labels = []

        pred_labels = []
        tot_acc = []
        out_data = []
        softmax_val = []

        pred_rank = []

        # loop = int((config.FILE_READ_LIMIT * config.NUM_OF_CLASS / config.BATCH_SIZE) - config.EPOCH)

        for root, subdirs, files in os.walk(path):
            root_split = root.split('/')
            if root_split[-1] == 'out':
                for file_name in files:
                    file_loc = os.path.join(root, file_name)
                    video_stream = cv2.VideoCapture(file_loc)
                    video = []
                    while(True):
                        ret, frame = video_stream.read()
                        if ret == False:
                            break
                        video.append(frame)
                    test_vids.append(video)

                    temp = [0] * config.NUM_OF_CLASS
                    temp[config.LABEL_MAP[root_split[-2]]] = 1
                    test_labels.append(temp)

        loop = len(test_vids) // config.BATCH_SIZE
        for i in range(loop):
            idx = i * config.BATCH_SIZE
            X = test_vids[idx : idx + config.BATCH_SIZE]
            Y = test_labels[idx : idx + config.BATCH_SIZE]

            pred, acc, out, softmax = session.run(
                fetches=[self.is_correct,
                         self.global_acc,
                         self.is_output,
                         self.softmax_val],
                feed_dict={
                    self.input: X,
                    self.label: Y,
                    self.learning_rate: .0})
            pred_labels.append(pred)
            tot_acc.append(acc)
            out_data.append(out)
            for j in range(config.BATCH_SIZE):
                softmax_val.append([softmax[j][out[j]], softmax[j][np.argmax(Y[j])]])
                rank = sorted(softmax[j], reverse=True)
                pred_rank.append(rank.index(softmax[j][np.argmax(Y[j])]))

        acc_val = np.mean(tot_acc)
        count_top1 = 0
        count_top2 = 0
        count_top3 = 0

        for val in pred_rank:
            if val < 1:
                count_top1 += 1
            if val < 2:
                count_top2 += 1
            if val < 3:
                count_top3 += 1

        print('[Top 1 ACC] ACC: %.4f' % acc_val)
        print('[Top 2 ACC] ACC: %.4f' % (count_top2 / len(test_vids)))
        print('[Top 3 ACC] ACC: %.4f' % (count_top3 / len(test_vids)))

        return acc_val
        # bar = Bar('Processing', max=len(test_vids), suffix='%(index)d/%(max)d - %(percent).1f%% - %(eta)ds')
        #
        # false_save_path = sample_save_path + '/sample_test'
        # out_data = np.reshape(out_data, -1)
        # softmax_val = np.reshape(softmax_val, (-1, 2))
        #
        # os.makedirs(false_save_path + '/original', exist_ok=True)
        # for j in range(config.NUM_OF_CLASS):
        #     for k in range(config.NUM_OF_CLASS):
        #         dir_name = false_save_path + '/' + config.REV_MAP[j] + '/' + config.REV_MAP[k]
        #         os.makedirs(dir_name, exist_ok=True)
        #
        # fout = open(false_save_path + '/pred.txt', 'w')
        # false_alarm = [0] * config.NUM_OF_CLASS ** 2
        # false_alarm = np.reshape(false_alarm, (config.NUM_OF_CLASS, config.NUM_OF_CLASS))
        #
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # for index, video in enumerate(test_vids):
        #     bar.next()
        #     k = np.argmax(test_labels[index])
        #     video_name = false_save_path + '/' + config.REV_MAP[k] \
        #                  + '/' + config.REV_MAP[out_data[index]] \
        #                  + '/' + str(softmax_val[index][0]) \
        #                  + '_' + str(softmax_val[index][1]) \
        #                  + '_' + str(index) + '.avi'
        #     writer = cv2.VideoWriter(video_name, fourcc, config.IMAGE_FRAMES, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        #     for frame in video:
        #         writer.write(frame)
        #     false_alarm[k][out_data[index]] += 1
        #     writer.release()
        #     video_name = false_save_path + '/original/' + str(index) + '.avi'
        #     writer = cv2.VideoWriter(video_name, fourcc, config.IMAGE_FRAMES, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT))
        #     for frame in video:
        #         writer.write(frame)
        #     writer.release()
        #
        # for i in range(config.NUM_OF_CLASS):
        #     fout.write('%s ' % config.REV_MAP[i])
        #     fout.write('\n')
        #     for j in range(config.NUM_OF_CLASS):
        #         fout.write('%d ' % false_alarm[i][j])
        #     fout.write('\n')
        #
        # bar.finish()
        # fout.close()

    def run_train(self, data_manager, model_save_path):
        session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        train_start_time = time.time()
        with tf.device('/cpu:0'), session.as_default():
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
            ckpt = tf.train.get_checkpoint_state(model_save_path)
            if ckpt and ckpt.model_checkpoint_path and not config.TRAIN_OVERWRITE:
                print(
                'Restore trained model from ' + ckpt.model_checkpoint_path)
                # model_name = model_save_path + '/tr-15700'
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print(
                'Create new model and overwrite previous files')

                os.makedirs(model_save_path, exist_ok=True)
                session.run(tf.local_variables_initializer())
                session.run(tf.global_variables_initializer())

            summary_writer = tf.summary.FileWriter(os.path.join(model_save_path, 'logs'))
            """
            Train your model iteratively
            """

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            epoch_acc = []
            test_acc = 0
            while session.run(self.global_step) < config.TRAIN_MAX_STEPS:

                step_index = int(session.run(self.global_step))

                lr = self.get_learning_rate(step_index=step_index)
                step_start_time = time.time()

                # read input data from data manager
                X, Y = data_manager.next_batch()
                X = np.reshape(X, [-1, config.IMAGE_FRAMES, config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.IMAGE_CHANNEL])
                Y = session.run(tf.one_hot(Y, depth=config.NUM_OF_CLASS))
                _, loss, acc, summary = session.run(
                    fetches=[
                        self.train_step,
                        self.global_loss,
                        self.global_acc,
                        self.summary
                    ],
                    feed_dict={
                        self.input: X,
                        self.label: Y,
                        self.learning_rate: lr
                    }
                )
                del(X)
                del(Y)

                # print useful logs in your console
                timecost = time.time() - step_start_time
                print(
                '[Step %5d] LR: %.5E, LOSS: %.5E, ACC: %.4f, Time: %.7f sec' % (step_index, lr, loss, acc, timecost))

                epoch_acc.append(acc)
                summary_writer.add_summary(summary, global_step=step_index)

                if step_index % config.EPOCH == 0:
                    print('[Epoch ACC: %.4f]' % np.mean(epoch_acc))

                    if step_index / config.EPOCH >= 90:
                        temp_acc_save = self.run_test(session)
                        if  test_acc < temp_acc_save:
                            test_acc = temp_acc_save
                            saver.save(session, os.path.join(model_save_path,'tr'), global_step=step_index)

                    elif step_index % config.MODEL_SAVE_INTERVAL == 0:
                        saver.save(session, os.path.join(model_save_path, 'tr'), global_step=step_index)

                    epoch_acc.clear()

            # export results
            # self.export_results(session, data_manager, True, sample_save_path)
            # self.export_results(session, test_data, False, sample_save_path)
            # self.run_test(session, sample_save_path)

            coord.request_stop()
            coord.join(threads)


        train_end_time = time.time()
        print(
        '--- All Training Done Successfully in %.7f seconds ----' % (train_end_time - train_start_time))
        summary_writer.add_graph(session.graph)
        summary_writer.close()
