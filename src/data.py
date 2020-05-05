import os

import numpy as np
try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf


class Data:
    def __init__(self, config):
        self.config = config

        self.train_triples = self.read_triples(os.path.join(self.config.in_path, 'train.txt'))
        self.valid_triples = self.read_triples(os.path.join(self.config.in_path, 'valid.txt'))
        self.test_triples = self.read_triples(os.path.join(self.config.in_path, 'test.txt'))

        self.ents = self.build_ents_list(self.train_triples + self.valid_triples + self.test_triples)
        self.rels = self.build_rels_list(self.train_triples + self.valid_triples + self.test_triples)

        # from here on, converting to id, working with id
        self.ent_id, self.id_ent, self.ents = self.build_id_dict(self.ents)
        self.rel_id, self.id_rel, self.rels = self.build_id_dict(self.rels)

        self.train_triples = self.build_triples_id(self.train_triples)
        self.valid_triples = self.build_triples_id(self.valid_triples)
        self.test_triples = self.build_triples_id(self.test_triples)

        # for correcting false negative in negative sampling
        self.known_htr_train = set(self.train_triples)
        self.known_htr_valid = set(self.valid_triples)
        self.known_htr_test = set(self.test_triples)
        self.known_htr_all = set(self.train_triples + self.valid_triples + self.test_triples)
        # for correcting false negative in 1-n scoring
        self.known_hr_t_train, self.known_tr_h_train = self.build_known_triples_dict(self.train_triples)
        self.known_hr_t_valid, self.known_tr_h_valid = self.build_known_triples_dict(self.valid_triples)
        self.known_hr_t_test, self.known_tr_h_test = self.build_known_triples_dict(self.test_triples)
        # for filtering false negative in test
        self.known_hr_t_all, self.known_tr_h_all = self.build_known_triples_dict(self.train_triples + self.valid_triples + self.test_triples)

        # statistics
        print('Num_ent: %i' % len(self.ents))
        print('Num_rel: %i' % len(self.rels))
        print('Num_train: %i' % len(self.train_triples))
        print('Num_valid: %i' % len(self.valid_triples))
        print('Num_test: %i' % len(self.test_triples))

        # for data sampling
        self.triple_sample = np.empty((self.config.batch_size + self.config.batch_size * self.config.neg_ratio, 3), dtype=np.int64, order='F')  # pre-allocate only once
        self.label_sample = np.empty((self.config.batch_size + self.config.batch_size * self.config.neg_ratio), dtype=np.float32)

    def read_triples(self, filepath):
        """
        :param filepath: file format: each line is 'h	r	t'
        :return: list of tuple [(h, t, r)], raw data.
        """
        triples = []
        with open(filepath, "r") as f:
            for line in f.readlines():
                h, r, t = line.split()  # note the order
                triples.append((h, t, r))
        return triples

    def build_ents_list(self, triples):
        """
        :param triples: (h, t, r)
        :return: sorted list of ent h, t
        """
        return sorted(list(set([e for (h, t, r) in triples for e in (h, t)])))

    def build_rels_list(self, triples):
        """
        :param triples: (h, t, r)
        :return: sorted list of rel r
        """
        return sorted(list(set([r for (h, t, r) in triples])))

    def build_id_dict(self, items):
        """
        Build bijective dictionary item to id.
        :param items: raw ents or rels
        :return: 2 dicts and items list in id format
        """
        item_id = {items[i]: i for i in range(len(items))}
        id_item = {i: items[i] for i in range(len(items))}
        items = [i for i in range(len(items))]
        return item_id, id_item, items

    def build_triples_id(self, triples):
        """
        :param triples: raw [(h, t, r)]
        :return: id [(h, t, r)]
        """
        return [(self.ent_id[h], self.ent_id[t], self.rel_id[r]) for (h, t, r) in triples]

    def build_known_triples_dict(self, triples):
        """
        Get known triples dict to filter positive/negative fast
        :param triples: (h, t, r)
        :return: (h, r): [t] and (t, r): [h]
        """
        known_hr_t = {}
        known_tr_h = {}
        for (h, t, r) in triples:
            if (h, r) not in known_hr_t:
                known_hr_t[(h, r)] = [t]
            elif t not in known_hr_t[(h, r)]:
                known_hr_t[(h, r)].append(t)

            if (t, r) not in known_tr_h:
                known_tr_h[(t, r)] = [h]
            elif h not in known_tr_h[(t, r)]:
                known_tr_h[(t, r)].append(h)

        return known_hr_t, known_tr_h

    def sampling(self, triples, idx, batch_size, triple_sample, label_sample):
        """
        Getting next batch.
        Remember to shuffle triples before each epoch.
        :param triples: data to sample from: such as self.train_triples
        :param idx: start index
        :param batch_size: batch size
        :param triple_sample: ref to output self.triple_sample ((htr)*(batch_size+batch_size*negratio))
        :param label_sample: ref to output self.label_sample (1*(batch_size+batch_size*negratio))
        :return: h, t, r, y: 4 arrays [size,].
        """
        if idx + batch_size > len(triples):  # handle smaller last batch
            batch_size = len(triples) - idx

        triple_sample[:batch_size] = triples[idx:idx + batch_size]  # copy positive triples
        label_sample[:batch_size] = 1.0

        neg_size = batch_size * self.config.neg_ratio
        if self.config.neg_ratio > 0:  # negative sampling in train
            rdm_entities = np.random.randint(0, len(self.ents), neg_size)  # pre-sample all negative entities
            triple_sample[batch_size:batch_size + neg_size, :] = np.tile(triple_sample[:batch_size], (self.config.neg_ratio, 1))  # pre-copy negative triples
            label_sample[batch_size:batch_size + neg_size] = 0.0
            rdm_choices = np.random.random(neg_size) < 0.5  # pre-sample choices head/tail
            for i in range(neg_size):
                if rdm_choices[i]:
                    triple_sample[batch_size + i, 1] = rdm_entities[i]  # corrupt tail
                else:
                    triple_sample[batch_size + i, 0] = rdm_entities[i]  # corrupt head

                if tuple(triple_sample[batch_size + i, :]) in self.known_htr_train:
                    label_sample[batch_size + i] = 1.0  # correcting false negative, rare negative will require large neg_ratio and many epoches to learn

        return triple_sample[:batch_size + neg_size, 0], triple_sample[:batch_size + neg_size, 1], triple_sample[:batch_size + neg_size, 2], label_sample[:batch_size + neg_size]
