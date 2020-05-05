import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


class Model(object):
    def __init__(self, data, config):
        self.data = data
        self.config = config

        with tf.name_scope("debug"):
            self.debug_def()

        with tf.name_scope("exp"):
            self.exp_def()

        with tf.name_scope("input"):
            self.input_def()

        with tf.name_scope("embedding"):
            self.embedding_def()

        with tf.name_scope("param"):
            self.param_def()

        with tf.name_scope("predict"):
            self.predict_def()

        with tf.name_scope("loss"):
            self.loss_def()

        with tf.name_scope("constraint"):
            self.constraint_def()

    def debug_def(self):
        """
        For general things: tracking, printing...
        """
        if self.config.debug:
            self.epoch = tf.placeholder_with_default(1, shape=())

    def exp_def(self):
        """
        For experiments: dropout...
        """
        self.is_training = tf.placeholder_with_default(False, shape=())

    def input_def(self):
        self.h = tf.placeholder(tf.int64, [None])
        self.t = tf.placeholder(tf.int64, [None])
        self.r = tf.placeholder(tf.int64, [None])
        self.y = tf.placeholder(tf.float32, [None])  # note that y == 1. and 0. for standard binary cross-entropy loss

    def embedding_def(self):
        """
        Define embedding matrices.
        """
        # Performance note:
        # gather slow; unstack/stack slow; reshape fast, transpose slow
        # gather seems to copy data: slower proportional to data
        # unstack/stack seems to also copy data: when unstack the whole num_ent embs will be very slow
        # unstack axis 1 is faster than axis 2, but still slower than separate list of embs with many embedding lookups

        self.ent_embs = tf.get_variable(name="ent_embs", shape=[len(self.data.ents), self.config.D, self.config.Ce],
                                        initializer=tf.keras.initializers.glorot_normal())
        self.rel_embs = tf.get_variable(name="rel_embs", shape=[len(self.data.rels), self.config.D, self.config.Cr],
                                        initializer=tf.keras.initializers.glorot_normal())

    def param_def(self):
        """
        Define weight vector used for combining trilinear products in our paper.
        """
        self.wv = tf.get_variable(name='wv', shape=[self.config.Ce * self.config.Ce * self.config.Cr],
                                  initializer=tf.keras.initializers.truncated_normal(mean=0.0, stddev=0.5))  # L2 reg manually in loss

    def compute_score(self, h, t, r):
        pass

    def predict_def(self):
        """
        Compute prediction, return score
        """
        self.predict_op = self.compute_score(self.h, self.t, self.r)  # note because argsort() ascending, need argmin, let score negative (-) before argsort

    def loss_def(self):
        """
        Compute loss
        - Forward compute score, with dropout, batchnorm, ...
        - Cross-entropy loss, with regularization loss, label smoothing, score scaling, ...
        :return:
        """
        print("h shape:")
        print(self.h.get_shape().as_list())  # (batch_size + batch_size * neg_ratio,)
        print("t shape:")
        print(self.t.get_shape().as_list())
        print("r shape:")
        print(self.r.get_shape().as_list())
        print("y shape:")
        print(self.y.get_shape().as_list())

        # Define score function for all positive triples and negative triples
        score = self.compute_score(self.h, self.t, self.r)

        # Define loss op to minimize with train_op in Config.
        # loss for one mini batch. Note: mean
        if self.config.loss_mode == 'cross-entropy':
            # directly binary cross-entropy, using tf function
            self.loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=score))
        elif self.config.loss_mode == 'softplus':
            # rewrite loss by softplus, change label from 1/0 to 1/-1
            # softplus is the same as bare binary cross-entropy: pushes score > 0 for y==1, < 0 for y==-1
            self.y = self.y * 2 - 1
            self.loss_op = tf.reduce_mean(tf.nn.softplus(- self.y * score))
        elif self.config.loss_mode == 'softmax-cross-entropy':
            # this is directly softmax cross-entropy, using tf function
            # softmax sum pos/neg class exponential and push up/down together
            # need to normalize labels to distribution, and specify full distribution axes
            y_distribution = self.y / tf.reduce_sum(self.y, axis=1, keepdims=True)
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_distribution, logits=score, axis=1))

        # emb l2 reg loss, only on active emb to reduce computation. Note: mean. Separately for better tuning lmbda.
        # FOR ADAPTIVE REG WEIGHT: count ent/rel frequency in batch, reduce mean unique ent/rel, multiply weight
        if self.config.lmbda_ent > 0:
            # FOR ADAPTIVE REG WEIGHT:
            unique, idx, count = tf.unique_with_counts(tf.concat([self.h, self.t], axis=0))
            weight = tf.cast(count / tf.reduce_sum(count), tf.float32)
            # FOR N3 REG:
            if self.config.reg_n3:
                self.loss_op += self.config.lmbda_ent * tf.reduce_mean(weight * tf.reduce_mean(tf.abs(tf.gather(self.ent_embs, unique)) ** 3, axis=[1, 2]))
            else:
                self.loss_op += self.config.lmbda_ent * tf.reduce_mean(weight * tf.reduce_mean(tf.gather(self.ent_embs, unique) ** 2, axis=[1, 2]))

        if self.config.lmbda_rel > 0:
            # FOR ADAPTIVE REG WEIGHT:
            unique, idx, count = tf.unique_with_counts(self.r)
            weight = tf.cast(count / tf.reduce_sum(count), tf.float32)
            # FOR N3 REG:
            if self.config.reg_n3:
                self.loss_op += self.config.lmbda_rel * tf.reduce_mean(weight * tf.reduce_mean(tf.abs(tf.gather(self.rel_embs, unique)) ** 3, axis=[1, 2]))
            else:
                self.loss_op += self.config.lmbda_rel * tf.reduce_mean(weight * tf.reduce_mean(tf.gather(self.rel_embs, unique) ** 2, axis=[1, 2]))

        # combinator params l2 reg loss manually. Note: mean.
        if self.config.lmbda_params > 0:
            self.loss_op += self.config.lmbda_params * tf.reduce_mean(self.wv ** 2)

    def constraint_def(self):
        """
        Constraint on embedding vector, such as unit norm, non negative
        """
        # Note that variable has raw shape, not including batchsize, constraint on axis 1 is each emb.
        # Also note that, for efficiency, only compute norm and updates embedding of ent/rel modified by gradient update
        if self.config.constraint == 'nonneg':
            def constraint_ent(x):
                return tf.keras.constraints.non_neg()(x)
            def constraint_rel(x):
                return tf.keras.constraints.non_neg()(x)
        elif self.config.constraint == 'unitnorm':
            def constraint_ent(x):
                return tf.keras.constraints.unit_norm(axis=self.config.constrain_axis_ent)(x)
            def constraint_rel(x):
                return tf.keras.constraints.unit_norm(axis=self.config.constrain_axis_rel)(x)
        elif self.config.constraint == 'unitnorm_nonneg':
            def constraint_ent(x):
                return tf.keras.constraints.unit_norm(axis=self.config.constrain_axis_ent)(tf.keras.constraints.non_neg()(x))
            def constraint_rel(x):
                return tf.keras.constraints.unit_norm(axis=self.config.constrain_axis_rel)(tf.keras.constraints.non_neg()(x))
        elif self.config.constraint == 'maxnorm':
            def constraint_ent(x):
                return tf.keras.constraints.max_norm(max_value=2, axis=self.config.constrain_axis_ent)(x)
            def constraint_rel(x):
                return tf.keras.constraints.max_norm(max_value=2, axis=self.config.constrain_axis_rel)(x)
        elif self.config.constraint == 'maxnorm_nonneg':
            def constraint_ent(x):
                return tf.keras.constraints.max_norm(max_value=2, axis=self.config.constrain_axis_ent)(tf.keras.constraints.non_neg()(x))
            def constraint_rel(x):
                return tf.keras.constraints.max_norm(max_value=2, axis=self.config.constrain_axis_rel)(tf.keras.constraints.non_neg()(x))
        else:
            return

        # assign and scatter_update seem to have no gradient, stop_gradient just to make sure
        self.constraint_all_embs_ops = []  # update all embs
        if 'ent' in self.config.to_constrain:
            self.constraint_all_embs_ops.append(tf.stop_gradient(
                tf.assign(self.ent_embs, constraint_ent(self.ent_embs))))
        if 'rel' in self.config.to_constrain:
            self.constraint_all_embs_ops.append(tf.stop_gradient(
                tf.assign(self.rel_embs, constraint_rel(self.rel_embs))))

        self.constraint_scatter_embs_ops = []  # only update active embs
        if 'ent' in self.config.to_constrain:
            self.constraint_scatter_embs_ops.append(tf.stop_gradient(
                tf.scatter_update(self.ent_embs, self.h, constraint_ent(self.hem))))
            self.constraint_scatter_embs_ops.append(tf.stop_gradient(
                tf.scatter_update(self.ent_embs, self.t, constraint_ent(self.tem))))
        if 'rel' in self.config.to_constrain:
            self.constraint_scatter_embs_ops.append(tf.stop_gradient(
                tf.scatter_update(self.rel_embs, self.r, constraint_rel(self.rem))))


class DistMult(Model):
    """
    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score = h0t0r0
        score = tf.reduce_sum(self.he[0] * self.te[0] * self.re[0], axis=1)
        return score


class CP(Model):
    """
    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score = h0t1r0
        score = tf.reduce_sum(self.he[0] * self.te[1] * self.re[0], axis=1)
        return score


class CPh(Model):
    """
    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score = h0t1r0 + h1t0r1
        score = tf.reduce_sum(self.he[0] * self.te[1] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[1] * self.te[0] * self.re[1], axis=1)
        return score


SimplE = CPh  # alias


class ComplEx(Model):
    """
    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score = h0t0r0 + h0t1r1 - h1t0r1 + h1t1r0
        score = tf.reduce_sum(self.he[0] * self.te[0] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[0] * self.te[1] * self.re[1], axis=1) \
                - tf.reduce_sum(self.he[1] * self.te[0] * self.re[1], axis=1) \
                + tf.reduce_sum(self.he[1] * self.te[1] * self.re[0], axis=1)
        return score


class Quaternion(Model):
    """
    Note that quaternion is very expressive, so it needs good regularization:
    - to restrict collapse/overfitting, entity and/or relation embeddings need to have unitnorm
    - always careful l2 reg on emb
    - for pure rotation, relation and/or entity embedding entries need to be unit quaternion

    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score
        score = tf.reduce_sum(self.he[0] * self.te[0] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[1] * self.te[1] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[2] * self.te[2] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[3] * self.te[3] * self.re[0], axis=1) \
                + tf.reduce_sum(self.he[0] * self.te[1] * self.re[1], axis=1) \
                - tf.reduce_sum(self.he[1] * self.te[0] * self.re[1], axis=1) \
                + tf.reduce_sum(self.he[2] * self.te[3] * self.re[1], axis=1) \
                - tf.reduce_sum(self.he[3] * self.te[2] * self.re[1], axis=1) \
                + tf.reduce_sum(self.he[0] * self.te[2] * self.re[2], axis=1) \
                - tf.reduce_sum(self.he[1] * self.te[3] * self.re[2], axis=1) \
                - tf.reduce_sum(self.he[2] * self.te[0] * self.re[2], axis=1) \
                + tf.reduce_sum(self.he[3] * self.te[1] * self.re[2], axis=1) \
                + tf.reduce_sum(self.he[0] * self.te[3] * self.re[3], axis=1) \
                + tf.reduce_sum(self.he[1] * self.te[2] * self.re[3], axis=1) \
                - tf.reduce_sum(self.he[2] * self.te[1] * self.re[3], axis=1) \
                - tf.reduce_sum(self.he[3] * self.te[0] * self.re[3], axis=1)
        return score


class TransE(Model):
    """
    Reimplement compute_score()
    """
    def compute_score(self, h, t, r):
        """
        Main logic: compute the score.
        Input: tensor in batch: (batch,) of indices (1-hot vector)
        :return: tensor in batch: score: (batch,)
        """
        print("D:")
        print(self.config.D)
        print(self.ent_embs.get_shape().as_list()[1])
        print("Ce:")
        print(self.config.Ce)
        print(self.ent_embs.get_shape().as_list()[2])
        print("Cr:")
        print(self.config.Cr)
        print(self.rel_embs.get_shape().as_list()[2])

        # Look up
        self.hem = tf.gather(self.ent_embs, h)  # (batch, d, Ce)
        self.tem = tf.gather(self.ent_embs, t)  # (batch, d, Ce)
        self.rem = tf.gather(self.rel_embs, r)  # (batch, d, Cr)
        print("he shape:")
        print(self.hem.get_shape().as_list())
        print("te shape:")
        print(self.tem.get_shape().as_list())
        print("re shape:")
        print(self.rem.get_shape().as_list())

        self.he = tf.unstack(self.hem, axis=2)  # [(batch, d),]
        self.te = tf.unstack(self.tem, axis=2)
        self.re = tf.unstack(self.rem, axis=2)

        # Compute score = -||h0+r0-t0||1|2
        score = - tf.reduce_sum(tf.abs(self.he[0] + self.re[0] - self.te[0]), axis=1)  # L1 usually better, also faster
        # score = - tf.reduce_sum(tf.abs(self.he[0] + self.re[0] - self.te[0])**2, axis=1)**(1/2)  # L2
        return score
