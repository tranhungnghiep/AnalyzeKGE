import os
import time

import numpy as np
import scipy as sp
import pandas as pd
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


class Experiment(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config

        tf.reset_default_graph()  # reset the tf graph in the current python thread, just to make sure
        self.graph = tf.Graph()  # construct a new graph to use, let old self.graph be gargabe collected
        self.sess = tf.Session(graph=self.graph, config=self.config.tfcfg)  # construct a new session manager deployed on this graph and following tfcfg, let old self.sess be garbage collected

        with self.graph.as_default():  # all ops runned by session in this scope are added to self.graph
            with self.sess.as_default():  # self.sess auto close after exit with block
                tf.set_random_seed(self.config.seed)

                with tf.variable_scope("model", reuse=None):
                    self.model = self.config.model_class(data=self.data, config=self.config)  # construct new model instance
                    self.lr = tf.placeholder_with_default(self.config.lr, shape=())  # learning rate placeholder for schedule
                    if self.config.opt_method.lower() == "adam":
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
                    else:
                        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
                    grads_and_vars = self.optimizer.compute_gradients(self.model.loss_op)  # computes gradients
                    self.train_op = self.optimizer.apply_gradients(grads_and_vars)  # applies gradients to minimize loss

                self.saver = tf.train.Saver()  # add save and load ops to the graph

                self.sess.run(tf.global_variables_initializer())  # initialize

                # TensorBoard: write the graph out after it was constructed and initialized
                self.tb_writer = tf.summary.FileWriter(self.config.tb_logdir)
                self.tb_writer.add_graph(self.graph)

    def train_step(self, h, t, r, y, epoch):
        feed_dict = {
            self.model.h: h,
            self.model.t: t,
            self.model.r: r,
            self.model.y: y,
            self.model.is_training: True
        }
        if self.config.debug:
            feed_dict[self.model.epoch] = epoch
        if self.config.lr_decay < 1:
            feed_dict[self.lr] = self.config.lr * (self.config.lr_decay ** epoch)  # lr decay after each epoch

        # Run loss and train ops
        # For old batchnorm, collect and run update_ops in each mini batch manually, remember name and auto reuse; do not use keras, not update running stats with custom training
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        _, _, loss = self.sess.run([update_ops, self.train_op, self.model.loss_op], feed_dict)  # train_op is run here

        # Constraint
        if self.config.constraint:
            self.sess.run(self.model.constraint_scatter_embs_ops, feed_dict)

        return loss

    def test_step(self, h, t, r):
        if self.config.batch_size_test == 0:
            batch_size_test = len(h)
        else:
            batch_size_test = self.config.batch_size_test

        predict_list = []
        for i in range(0, len(h), batch_size_test):
            feed_dict = {
                self.model.h: h[i: i + batch_size_test],
                self.model.t: t[i: i + batch_size_test],
                self.model.r: r[i: i + batch_size_test]
            }
            # list of predict for each test mini batch
            predict_list.append(self.sess.run(self.model.predict_op, feed_dict))

        predict = np.concatenate(predict_list, axis=0)  # (batch,)
        assert len(h) == predict.shape[0]  # make sure no error in batching

        return predict

    def run(self, check_period=5, early_stop=False, patience=20):
        with self.graph.as_default():
            with self.sess.as_default():

                # Constraint
                if self.config.constraint:  # constraint all embs when start
                    self.sess.run(self.model.constraint_all_embs_ops)

                best_valid_f_mrr = 0.0
                best_epoch = 0
                violation = 0
                for epoch in range(1, self.config.max_epoch + 1):  # 1 to == max_epoch
                    epoch_time = time.time()  # in seconds
                    loss = 0.0

                    np.random.shuffle(self.data.train_triples)  # shuffle every epoch
                    batch_size = self.config.batch_size
                    for idx in range(0, len(self.data.train_triples), batch_size):
                        h, t, r, y = self.data.sampling(self.data.train_triples, idx, batch_size, self.data.triple_sample, self.data.label_sample)
                        loss += self.train_step(h, t, r, y, epoch)

                    if self.config.log:
                        print(epoch)
                        print('Epoch time (s): ' + str(time.time() - epoch_time))
                        print('Loss: %E' % (loss / (len(self.data.train_triples) / self.config.batch_size)))

                    if self.config.out_path and (self.config.export_steps != 0 and epoch % self.config.export_steps == 0):  # save model after each export_steps
                        self.save_tensorflow(file_name=self.config.model_file_name, epoch=epoch)

                    if epoch % check_period == 0:  # meet check period, not first epoch, maybe last epoch
                        self.test(self.data.train_triples[:1000], test_case='train')
                        valid_f_mrr = self.test(self.data.valid_triples, test_case='valid')[6]
                        self.test(self.data.test_triples, test_case='test')
                        if valid_f_mrr > best_valid_f_mrr:
                            best_valid_f_mrr = valid_f_mrr
                            best_epoch = epoch
                            violation = 0
                            print('***New best valid f_mrr %.3f at epoch %i.***\n\n' % (best_valid_f_mrr, best_epoch))
                        else:
                            violation += 1
                        if early_stop:
                            if violation == 0:
                                if self.config.out_path:
                                    print('Saving model at epoch %i.\n' % best_epoch)
                                    self.save_tensorflow(file_name=self.config.model_file_name, epoch=best_epoch)
                            elif violation > patience:
                                break

                print('Finish training. Best valid f_mrr %.3f at epoch %i.\n\n' % (best_valid_f_mrr, best_epoch))

                if early_stop:
                    if epoch < self.config.max_epoch:
                        print('Early stop by filtered MRR on valid.')
                    else:
                        print('Stop by last epoch.')
                    print('Test at best epoch. Loading saved model at epoch %i.' % best_epoch)
                    self.restore_tensorflow(file_name=self.config.model_file_name, epoch=best_epoch)
                    self.test(self.data.train_triples[:1000], test_case='train')
                    self.test(self.data.valid_triples, test_case='valid')
                    self.test(self.data.test_triples, test_case='test')

                if self.config.out_path:  # save final (best if early_stop) model
                    self.save_tensorflow()

    def test(self, triples=None, test_case='test', is_verbose=False):
        """
        Test link prediction purely in python
        :param triples: [(h, t, r)]
        :param test_case: name of test case for printing: test, valid, train
        :return: r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10
        """
        base_time = time.time()  # in seconds
        with self.graph.as_default():
            with self.sess.as_default():
                if triples is None:
                    triples = self.data.test_triples
                total = len(triples)

                # Compute the rank of every triple, each triple has tail rank and head rank:
                raw_ranks = np.zeros(2 * total)  # init zero to used in on progress report
                ranks = np.zeros(2 * total)

                triples_t = np.empty((len(self.data.ents), 3), dtype=np.int64, order='F')  # empty: not init faster, order F is column wise, passing to tf faster
                triples_h = np.empty((len(self.data.ents), 3), dtype=np.int64, order='F')
                triples_t[:, 1] = np.arange(len(self.data.ents))  # pre vary tail 0->num_ent, note format is (h, t, r)
                triples_h[:, 0] = np.arange(len(self.data.ents))  # pre vary head 0->num_ent, note format is (h, t, r)

                def eval_t(h, r):
                    triples_t[:, [0, 2]] = (h, r)  # simple tile h r by broadcast, never change t
                    return self.test_step(triples_t[:, 0], triples_t[:, 1], triples_t[:, 2])

                def eval_h(t, r):
                    triples_h[:, [1, 2]] = (t, r)  # simple tile t r by broadcast, never change h
                    return self.test_step(triples_h[:, 0], triples_h[:, 1], triples_h[:, 2])

                for i, (h, t, r) in enumerate(triples):
                    # Computing objects ranks
                    res_t = eval_t(h, r)  # note positive score, larger is better
                    raw_ranks[i] = 1 + np.sum(res_t > res_t[t])  # strictly larger
                    ranks[i] = raw_ranks[i] - np.sum(res_t[self.data.known_hr_t_all[(h, r)]] > res_t[t])

                    # Computing subjects ranks
                    res_h = eval_h(t, r)
                    raw_ranks[total + i] = 1 + np.sum(res_h > res_h[h])
                    ranks[total + i] = raw_ranks[total + i] - np.sum(res_h[self.data.known_tr_h_all[(t, r)]] > res_h[h])

                    if is_verbose:
                        if (i < 10) or (i % 100 == 0 and total < 10 ** 3) or (i % 10 ** 3 == 0 and total < 10 ** 4) or (i % 10 ** 4 == 0):
                            print(i)
                            raw_ranks_temp = raw_ranks[raw_ranks > 0]  # only take non-zero ranks
                            ranks_temp = ranks[ranks > 0]
                            r_mr = np.mean(raw_ranks_temp)
                            r_mrr = np.mean(1.0 / raw_ranks_temp)
                            r_hit1 = np.mean(raw_ranks_temp <= 1)
                            r_hit3 = np.mean(raw_ranks_temp <= 3)
                            r_hit10 = np.mean(raw_ranks_temp <= 10)

                            f_mr = np.mean(ranks_temp)
                            f_mrr = np.mean(1.0 / ranks_temp)
                            f_hit1 = np.mean(ranks_temp <= 1)
                            f_hit3 = np.mean(ranks_temp <= 3)
                            f_hit10 = np.mean(ranks_temp <= 10)

                            print('Computing:\tmr\tmrr\thit1\thit3\thit10\n'
                                  'Raw:\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n'
                                  'Filter:\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n'
                                  % (r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10))

        r_mr = np.mean(raw_ranks)
        r_mrr = np.mean(1.0 / raw_ranks)
        r_hit1 = np.mean(raw_ranks <= 1)
        r_hit3 = np.mean(raw_ranks <= 3)
        r_hit10 = np.mean(raw_ranks <= 10)

        f_mr = np.mean(ranks)
        f_mrr = np.mean(1.0 / ranks)
        f_hit1 = np.mean(ranks <= 1)
        f_hit3 = np.mean(ranks <= 3)
        f_hit10 = np.mean(ranks <= 10)

        print('='*50)
        print('Test result on %s (%i triples, %f seconds):\n'
              'Metric\tmr\tmrr\thit1\thit3\thit10\n'
              'Raw:\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\n'
              'Filter:\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f'
              % (test_case, total, (time.time() - base_time), r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10))
        print('='*50 + '\n')

        return r_mr, r_mrr, r_hit1, r_hit3, r_hit10, f_mr, f_mrr, f_hit1, f_hit3, f_hit10

    def show_link_prediction(self, h, t, r, raw=True):
        """
        Show top tail and top head predictions for this triple (h, t, r)
        """
        if raw:
            h = self.data.ent_id[h]
            t = self.data.ent_id[t]
            r = self.data.rel_id[r]
        with self.graph.as_default():
            with self.sess.as_default():
                # batch predict score, argsort get the index
                top_tid = (-self.test_step(np.array([h] * len(self.data.ents)), np.arange(len(self.data.ents)), np.array([r] * len(self.data.ents))).reshape(-1)).argsort()  # argsort() return indices of values sorted ascending, -predict makes score descending, the indices is also ent id.
                top_hid = (-self.test_step(np.arange(len(self.data.ents)), np.array([t] * len(self.data.ents)), np.array([r] * len(self.data.ents))).reshape(-1)).argsort()  # argsort() return indices of values sorted ascending, -predict makes score descending, the indices is also ent id.
        print('Prediction for (%i, %i, %i) : (%s, %s, %s)' % (h, t, r, self.data.id_ent[h], self.data.id_ent[t], self.data.id_rel[r]))
        print('Top predicted tail:')
        [print('%i. %i : %s' % ((top + 1), id, self.data.id_ent[id])) for top, id in enumerate(top_tid[:10])]
        print('Top predicted head:')
        [print('%i. %i : %s' % ((top + 1), id, self.data.id_ent[id])) for top, id in enumerate(top_hid[:10])]

    def save_tensorflow(self, file_path=None, file_name=None, epoch=None):
        """
        Save full model in self.sess by tf.train.Saver, all vars must have been initialized.
        Default save path = self.config.out_path/self.config.model_file_name
        :param file_path: save path = file_path, prefix of full saved file path
        :param file_name: save path = outpath/file_name when file_path is None
        :param epoch: appended to save path to differentiate epoch
        """
        if file_path is None:
            if file_name is not None:  # save path = outpath/file_name
                file_path = os.path.join(self.config.out_path, file_name)
                if epoch is not None:  # append epoch
                    file_path = '%s-%08d' % (file_path, epoch)
            else:  # default save path = out_path/model_file_name
                file_path = os.path.join(self.config.out_path, self.config.model_file_name)

        with self.graph.as_default():
            with self.sess.as_default():
                self.saver.save(self.sess, file_path)

    def restore_tensorflow(self, file_path=None, file_name=None, epoch=None, reuse_session=True):
        """
        Restore full model to self.graph by tf.train.Saver.
        Note that this will destroy current default session and graph, and overwrite everything by saved data.
        Default save path = self.config.out_path/self.config.model_file_name
        :param file_path: save path = file_path, prefix of full saved file path
        :param file_name: save path = outpath/file_name when file_path is None
        :param epoch: appended to save path to differentiate epoch
        :param reuse_session: reuse current session and graph, keep reference to all variables, only change values; if restore to new session and graph, will need to access variables by name
        """
        if file_path is None:
            if file_name is not None:  # save path = outpath/file_name
                file_path = os.path.join(self.config.out_path, file_name)
                if epoch is not None:  # append epoch
                    file_path = '%s-%08d' % (file_path, epoch)
            else:  # default save path = out_path/model_file_name
                file_path = os.path.join(self.config.out_path, self.config.model_file_name)

        if reuse_session:  # restore to current graph
            with self.graph.as_default():  # all ops runned by session in this scope are added to self.graph
                with self.sess.as_default():  # self.sess auto close after exit with block
                    self.saver.restore(self.sess, file_path)  # load saved data to default graph self.graph
        else:  # restore to clean graph
            tf.reset_default_graph()  # reset the tf graph in the current python thread, just to make sure
            self.graph = tf.Graph()  # construct a new graph to use, let old self.graph be gargabe collected
            self.sess = tf.Session(graph=self.graph, config=self.config.tfcfg)  # construct a new session manager deployed on this graph and following tfcfg, let old self.sess be garbage collected
            with self.graph.as_default():  # all ops runned by session in this scope are added to self.graph
                with self.sess.as_default():  # self.sess auto close after exit with block
                    self.saver = tf.train.import_meta_graph(file_path + '.meta') # load saved graph to default graph self.graph, return new saver
                    self.saver.restore(self.sess, file_path)  # load saved data to default graph self.graph
