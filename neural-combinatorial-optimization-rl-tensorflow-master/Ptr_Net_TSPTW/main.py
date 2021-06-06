#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Ptr_Net_TSPTW.dataset import DataGenerator
from Ptr_Net_TSPTW.actor import Actor
from Ptr_Net_TSPTW.config import get_config, print_config
from Ptr_Net_TSPTW.multy import do_multy
from Ptr_Net_TSPTW.rand import do_rand


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()

    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config)

    # Saver to save & restore all the variables.
    variables_to_save = [v for v in tf.global_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(var_list=variables_to_save, keep_checkpoint_every_n_hours=1.0)

    predictions = []
    time_used = []
    task_priority = []
    ns_ = []

    with tf.Session() as sess:
        # Run initialize op
        sess.run(tf.global_variables_initializer())

        # Restore variables from disk.
        if config.restore_model is True:
            saver.restore(sess, config.restore_from)
            print("Model restored.")

        training_set = DataGenerator(config)

        # 训练
        if not config.inference_mode:

            print("Starting training...")
            for i in tqdm(range(config.nb_epoch)):
                # Get feed dict
                input_batch = training_set.train_batch()
                feed = {actor.input_: input_batch}
                # Forward pass & train step

                result, time_use, task_priority_sum, ns_prob, train_step1, train_step2 = sess.run(
                    [actor.reward, actor.time_use, actor.task_priority_sum, actor.ns_prob,
                     actor.train_step1, actor.train_step2],
                    feed_dict=feed)

                predictions.append(np.mean(time_use + task_priority_sum + ns_prob))
                time_used.append(np.mean(time_use))
                task_priority.append(np.mean(task_priority_sum))
                ns_.append(np.mean(ns_prob))

                if i % 100 == 0 and i != 0:
                    print('after '+str(i)+' rounds training, reward: ' + str(predictions[-1]) + ' time: ' + str(time_used[-1]) + ' task_priority: ' + str(task_priority[-1]) + ' ns_prob: ' + str(ns_[-1]))

                # # Save the variables to disk
                # if i % 1000 == 0 and i != 0:
                #     save_path = saver.save(sess, config.save_to)
                #     print("Model saved in file: %s" % save_path)

            print("Training COMPLETED !")
            # save_path = saver.save(sess, config.save_to)
            # print("Model saved in file: %s" % save_path)

        # 测试
        else:
            input_batch = training_set.train_batch()
            feed = {actor.input_: input_batch}
            print(feed)
            result, time_use, task_priority_sum, ns_prob, _, _ = sess.run(
                [actor.reward, actor.time_use, actor.task_priority_sum, actor.ns_prob,
                 actor.train_step1, actor.train_step2],
                feed_dict=feed)

            print('reward: ', np.mean(result))
            print('运行时间: ', np.mean(time_use))
            print('优先级: ', np.mean(task_priority_sum))
            print('超时率: ', np.mean(ns_prob))

    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    plt.rcParams['savefig.dpi'] = 400  # 图片像素
    plt.rcParams['figure.dpi'] = 400  # 分辨率

    fig = plt.figure()
    plt.plot(list(range(len(predictions))), predictions, c='red', label=u'指针网络')
    plt.title(u"效果曲线")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(time_used))), time_used, c='red', label=u'指针网络')
    plt.title(u"目标1：运行时间")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(task_priority))), task_priority, c='red', label=u'指针网络')
    plt.title(u"目标2：任务优先级")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    fig = plt.figure()
    plt.plot(list(range(len(ns_))), ns_, c='red', label=u'指针网络')
    plt.title(u"目标3：超时率")
    plt.xlabel('轮数')
    plt.legend()
    fig.show()

    # rand_result, rand_time_result, rand_task_priority_result, rand_ns_result = do_rand(input_batch, 0)
    # greed_result, greed_1_result, greed_2_result, greed_3_result = do_rand(input_batch, 1)
    # multy_result, multy_1_result, multy_2_result, multy_3_result = do_multy(input_batch)
    #
    # print('task:', config.max_length)
    # print('gen_num:', config.gen_num)
    # print('nb_epoch:', config.nb_epoch)
    # print('ptr')
    # print('综合效果', np.mean(predictions[-10:]))
    # print('目标1：运行时间', np.mean(time_used[-10:]))
    # print('目标2：任务优先级', np.mean(task_priority[-10:]))
    # print('目标3：超时率', np.mean(ns_[-10:]))
    # print('greed')
    # print('综合效果', np.mean(greed_result[-10:]))
    # print('目标1：运行时间', np.mean(greed_1_result[-10:]))
    # print('目标2：任务优先级', np.mean(greed_2_result[-10:]))
    # print('目标3：超时率', np.mean(greed_3_result[-10:]))
    # print('rand')
    # print('综合效果', np.mean(rand_result[-10:]))
    # print('目标1：运行时间', np.mean(rand_time_result[-10:]))
    # print('目标2：任务优先级', np.mean(rand_task_priority_result[-10:]))
    # print('目标3：超时率',np.mean(rand_ns_result[-10:]))
    # print('multy')
    # print('综合效果', np.mean(multy_result[-10:]))
    # print('目标1：运行时间', np.mean(multy_1_result[-10:]))
    # print('目标2：任务优先级', np.mean(multy_2_result[-10:]))
    # print('目标3：超时率', np.mean(multy_3_result[-10:]))


if __name__ == "__main__":
    main()
