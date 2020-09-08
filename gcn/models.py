from gcn.layers import GraphConvolution, FullyConnected
from gcn.metrics import masked_accuracy, masked_softmax_cross_entropy,masked_information_entropy
import tensorflow as tf
import numpy as np
from copy import copy
from gcn.utils import recursive_map

def cosine_similarity(inputs, output_dim, placeholders, k_ratio):

    # get the number of unlabeled data needed for adjusting centroid
    assert k_ratio <= 1 and k_ratio >= 0, 'k_ratio is greater than 1 or less than 0'
    k = tf.math.multiply(tf.reduce_sum(tf.cast(placeholders['unlabels_mask'], tf.float32)), k_ratio)
    k = tf.cast(k, tf.int32)

    label = tf.cast(placeholders['labels'], tf.float32)
    placeholders['labels_mask'].set_shape([None])
    placeholders['unlabels_mask'].set_shape([None])

    # inputs = tf.Variable(inputs,dtype=tf.float32,validate_shape=False)
    # inputs.set_shape([None])

    # training_set, query, unlabeled = tf.split(inputs, [int(idx * fraction), idx - int(idx * fraction), -1], 0)
    training_set = tf.boolean_mask(inputs, placeholders['labels_mask'])
    unlabeled = tf.boolean_mask(inputs, placeholders['unlabels_mask'])
    training_label = tf.boolean_mask(label, placeholders['labels_mask'])

    # Average vector based on the training set (input)
    total = tf.linalg.matmul(tf.transpose(training_set), training_label)
    weight = tf.math.reciprocal(tf.reduce_sum(label, 0))
    protos = tf.transpose(tf.linalg.matmul(total, tf.diag(weight)))

    if k_ratio != 0:

        unlabeled_prob = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.l2_normalize(unlabeled, dim=1), 1), tf.nn.l2_normalize(protos, dim=1)), 2)
        prob_top, prob_idx = tf.nn.top_k(unlabeled_prob, k, sorted=False)
        prob_sm = tf.nn.softmax(prob_top)
        prob_row_idx = tf.tile(tf.range(tf.shape(unlabeled_prob)[0])[:, tf.newaxis], (1, k))
        scatter_idx = tf.stack([prob_row_idx, prob_idx], axis=-1)
        unlabeled_prob = tf.transpose(tf.scatter_nd(scatter_idx, prob_top, tf.shape(unlabeled_prob)))
    

        unlabeled_weight = tf.math.add(tf.reduce_sum(unlabeled_prob, 0), tf.reduce_sum(training_label, 0))
        unlabeled_weight = tf.linalg.matmul(unlabeled_prob, tf.diag(tf.math.reciprocal(unlabeled_weight)))
        unlabeled_weight = tf.unstack(tf.expand_dims(tf.transpose(tf.unstack(unlabeled_weight, axis=1)), axis=1), axis=2)
        unlabeled_difference = tf.math.subtract(tf.expand_dims(unlabeled, axis=1), tf.expand_dims(protos, axis=0))
        unlabeled_difference = tf.transpose(tf.unstack(unlabeled_difference, axis=1), perm=[0, 2, 1])
        change = tf.reshape(tf.linalg.matmul(unlabeled_difference, unlabeled_weight), [output_dim, output_dim])
        protos = tf.math.add(change, protos)

    logits = tf.reduce_sum(tf.multiply(tf.expand_dims(tf.nn.l2_normalize(inputs, dim=1), 1), tf.nn.l2_normalize(protos, dim=1)), 2)

    return logits

def prototypical_net(inputs, output_dim, placeholders, k_ratio):

    # get the number of unlabeled data needed for adjusting centroid
    assert k_ratio <= 1 and k_ratio >= 0, 'k_ratio is greater than 1 or less than 0'
    k = tf.math.multiply(tf.reduce_sum(tf.cast(placeholders['unlabels_mask'], tf.float32)), k_ratio)
    k = tf.cast(k, tf.int32)

    label = tf.cast(placeholders['labels'], tf.float32)
    placeholders['labels_mask'].set_shape([None])
    placeholders['unlabels_mask'].set_shape([None])

    # inputs = tf.Variable(inputs,dtype=tf.float32,validate_shape=False)
    # inputs.set_shape([None])

    # training_set, query, unlabeled = tf.split(inputs, [int(idx * fraction), idx - int(idx * fraction), -1], 0)
    training_set = tf.boolean_mask(inputs, placeholders['labels_mask'])
    unlabeled = tf.boolean_mask(inputs, placeholders['unlabels_mask'])
    training_label = tf.boolean_mask(label, placeholders['labels_mask'])

    # Average vector based on the training set (input)
    total = tf.linalg.matmul(tf.transpose(training_set), training_label)
    weight = tf.math.reciprocal(tf.reduce_sum(label, 0))
    protos = tf.transpose(tf.linalg.matmul(total, tf.diag(weight)))

    
    # Average vector based on the training set and unlabeled set (input & unlabeled)
    unlabeled_prob = tf.reduce_sum(-tf.square(tf.math.subtract(tf.expand_dims(unlabeled, axis=1),
                                                               tf.expand_dims(protos, axis=0))),2)
    unlabeled_prob = tf.transpose(tf.nn.softmax(unlabeled_prob))
    
    # select the top k probs for each class as the unlabeled data
    prob_top, prob_idx = tf.nn.top_k(unlabeled_prob, k, sorted=False)
    prob_sm = tf.nn.softmax(prob_top)
    prob_row_idx = tf.tile(tf.range(tf.shape(unlabeled_prob)[0])[:, tf.newaxis], (1, k))
    scatter_idx = tf.stack([prob_row_idx, prob_idx], axis=-1)
    unlabeled_prob = tf.transpose(tf.scatter_nd(scatter_idx, prob_top, tf.shape(unlabeled_prob)))
    

    unlabeled_weight = tf.math.add(tf.reduce_sum(unlabeled_prob, 0), tf.reduce_sum(training_label, 0))
    unlabeled_weight = tf.linalg.matmul(unlabeled_prob, tf.diag(tf.math.reciprocal(unlabeled_weight)))
    unlabeled_weight = tf.unstack(tf.expand_dims(tf.transpose(tf.unstack(unlabeled_weight, axis=1)), axis=1), axis=2)
    unlabeled_difference = tf.math.subtract(tf.expand_dims(unlabeled, axis=1), tf.expand_dims(protos, axis=0))
    unlabeled_difference = tf.transpose(tf.unstack(unlabeled_difference, axis=1), perm=[0, 2, 1])
    change = tf.reshape(tf.linalg.matmul(unlabeled_difference, unlabeled_weight), [output_dim, output_dim])
    protos = tf.math.add(change, protos)
    

    # compute distance for all data in inputs based on protos
    logits = tf.reduce_sum(-tf.square(tf.math.subtract(tf.expand_dims(inputs, axis=1),tf.expand_dims(protos, axis=0))),2)

    return logits

def l1_distance(inputs, output_dim, placeholders, k_ratio):

    # get the number of unlabeled data needed for adjusting centroid
    assert k_ratio <= 1 and k_ratio >= 0, 'k_ratio is greater than 1 or less than 0'
    k = tf.math.multiply(tf.reduce_sum(tf.cast(placeholders['unlabels_mask'], tf.float32)), k_ratio)
    k = tf.cast(k, tf.int32)

    label = tf.cast(placeholders['labels'], tf.float32)
    placeholders['labels_mask'].set_shape([None])
    placeholders['unlabels_mask'].set_shape([None])

    # inputs = tf.Variable(inputs,dtype=tf.float32,validate_shape=False)
    # inputs.set_shape([None])

    # training_set, query, unlabeled = tf.split(inputs, [int(idx * fraction), idx - int(idx * fraction), -1], 0)
    training_set = tf.boolean_mask(inputs, placeholders['labels_mask'])
    unlabeled = tf.boolean_mask(inputs, placeholders['unlabels_mask'])
    training_label = tf.boolean_mask(label, placeholders['labels_mask'])

    # Average vector based on the training set (input)
    total = tf.linalg.matmul(tf.transpose(training_set), training_label)
    weight = tf.math.reciprocal(tf.reduce_sum(label, 0))
    protos = tf.transpose(tf.linalg.matmul(total, tf.diag(weight)))

    
    # Average vector based on the training set and unlabeled set (input & unlabeled)
    unlabeled_prob = tf.reduce_sum(-tf.abs(tf.math.subtract(tf.expand_dims(unlabeled, axis=1),
                                                               tf.expand_dims(protos, axis=0))),2)
    unlabeled_prob = tf.transpose(tf.nn.softmax(unlabeled_prob))
    
    # select the top k probs for each class as the unlabeled data
    prob_top, prob_idx = tf.nn.top_k(unlabeled_prob, k, sorted=False)
    prob_sm = tf.nn.softmax(prob_top)
    prob_row_idx = tf.tile(tf.range(tf.shape(unlabeled_prob)[0])[:, tf.newaxis], (1, k))
    scatter_idx = tf.stack([prob_row_idx, prob_idx], axis=-1)
    unlabeled_prob = tf.transpose(tf.scatter_nd(scatter_idx, prob_top, tf.shape(unlabeled_prob)))
    

    unlabeled_weight = tf.math.add(tf.reduce_sum(unlabeled_prob, 0), tf.reduce_sum(training_label, 0))
    unlabeled_weight = tf.linalg.matmul(unlabeled_prob, tf.diag(tf.math.reciprocal(unlabeled_weight)))
    unlabeled_weight = tf.unstack(tf.expand_dims(tf.transpose(tf.unstack(unlabeled_weight, axis=1)), axis=1), axis=2)
    unlabeled_difference = tf.math.subtract(tf.expand_dims(unlabeled, axis=1), tf.expand_dims(protos, axis=0))
    unlabeled_difference = tf.transpose(tf.unstack(unlabeled_difference, axis=1), perm=[0, 2, 1])
    change = tf.reshape(tf.linalg.matmul(unlabeled_difference, unlabeled_weight), [output_dim, output_dim])
    protos = tf.math.add(change, protos)
    

    # compute distance for all data in inputs based on protos
    logits = tf.reduce_sum(-tf.abs(tf.math.subtract(tf.expand_dims(inputs, axis=1),tf.expand_dims(protos, axis=0))),2)

    return logits


class IGCN(object):
    def __init__(self, model_config, placeholders, is_gcn, k_ratio, weight, method, input_dim):
        self.model_config = model_config
        self.name = model_config['name']
        if not self.name:
            self.name = self.__class__.__name__.lower()
        self.logging = True if self.model_config['logdir'] else False

        self.vars = {}
        self.layers = []
        self.activations = []
        self.act = [tf.nn.relu for i in range(len(self.model_config['connection']))]
        self.act[-1] = lambda x: x

        self._placeholders = placeholders
        self.assign_data = []

        def wrap_placeholder(placeholder):
            if isinstance(placeholder, tf.Tensor):
                dtype = placeholder.dtype
                shape = placeholder.shape
                tensor = tf.Variable(0,
                                     name=placeholder.name.split(':')[0], trainable=False, dtype=placeholder.dtype,
                                     expected_shape=shape, validate_shape=False, collections=[])
                tensor.set_shape(shape)
                self.assign_data.append(tf.assign(tensor, placeholder, validate_shape=False))
                return tensor
            elif isinstance(placeholder, tf.SparseTensor):
                return tf.SparseTensor(indices=wrap_placeholder(placeholder.indices),
                                       values=wrap_placeholder(placeholder.values),
                                       dense_shape=wrap_placeholder(placeholder.dense_shape))

        self.placeholders = recursive_map(placeholders, wrap_placeholder)
        self.assign_data = tf.group(self.assign_data)

        # self.placeholders = placeholders
        self.k_ratio = k_ratio
        self.is_gcn = is_gcn
        self.inputs = self.placeholders['features']
        self.inputs._my_input_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = self.placeholders['labels'].get_shape().as_list()[1]
        self.outputs = None
        self.outs = None
        self.outs_softmax =None
        self.mo_accuarcy = 0
        self.outputs_weight = weight
        self.global_step = None
        self.loss = 0
        self.outs_for_graph = 0
        self.cross_entropy_loss=0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None
        self.summary = None
        self.optimizer = model_config['optimizer'](learning_rate=self.model_config['learning_rate'])
        self.method = method

        self.build()
        return

    def build(self):
        layer_type = list(map(
            lambda x: {'c': GraphConvolution, 'f': FullyConnected}.get(x),
            self.model_config['connection']))
        layer_size = copy(self.model_config['layer_size'])
        layer_size.insert(0, self.input_dim)
        layer_size.append(self.output_dim)
        sparse = isinstance(self.placeholders['features'], tf.SparseTensor)
        with tf.name_scope(self.name):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.activations.append(self.inputs)
            for output_dim, layer_cls, act, conv_config in zip(layer_size[1:], layer_type, self.act,
                                                               self.model_config['conv_config']):
                # create Variables
                self.layers.append(layer_cls(input=self.activations[-1],
                                             output_dim=output_dim,
                                             placeholders=self.placeholders,
                                             act=act,
                                             dropout=True,
                                             sparse_inputs=sparse,
                                             logging=self.logging,
                                             conv_config=conv_config))
                sparse = False
                # Build sequential layer model
                hidden = self.layers[-1]()  # build the graph, give layer inputs, return layer outpus
                self.activations.append(hidden)

            self.outputs = self.activations[-1]

        if self.is_gcn and self.method == 'cos':
            self.outs = cosine_similarity(self.outputs, self.output_dim, self.placeholders, self.k_ratio)
            print('cos')
        elif self.is_gcn and self.method == 'l2':
            self.outs = prototypical_net(self.outputs, self.output_dim, self.placeholders, self.k_ratio)
            print('l2')
        elif self.is_gcn and self.method == 'l1':
            self.outs = l1_distance(self.outputs, self.output_dim, self.placeholders, self.k_ratio)
            print('l1')

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars.update({var.op.name: var for var in variables})

        # Build metrics
        with tf.name_scope('predict'):
            self._predict()
        with tf.name_scope('loss'):
            self._loss()
        tf.summary.scalar('loss', self.loss)
        with tf.name_scope('accuracy'):
            self._accuracy()
            self._accuracy_of_class()
        tf.summary.scalar('accuracy', self.accuracy)

        self.opt_op = self.optimizer.minimize(self.cross_entropy_loss, global_step=self.global_step)
        self.summary = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)

    def _predict(self):
        self.prediction = tf.nn.softmax(self.outputs)

    def _loss(self):
        # Weight decay loss
        for layer in self.layers:
            for var in layer.vars.values():
                self.cross_entropy_loss += self.model_config['weight_decay'] * tf.nn.l2_loss(var)

        
        self.cross_entropy_loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])
        if self.is_gcn:
            self.cross_entropy_loss += self.outputs_weight* masked_softmax_cross_entropy(self.outs, self.placeholders['labels'], self.placeholders['labels_mask'])
    def _laplacian_regularization(self, graph_signals):
        self.lapla_reg = tf.sparse_tensor_dense_matmul(self.placeholders['laplacian'], graph_signals)
        self.lapla_reg = tf.matmul(tf.transpose(graph_signals), self.lapla_reg)
        self.lapla_reg = tf.trace(self.lapla_reg)

    def _accuracy(self):

        if self.is_gcn:
            #self.merged = tf.math.add(tf.math.scalar_mul(self.outputs_weight, self.outputs),
                                      #tf.math.scalar_mul(self.outs_weight, self.outs))
            self.outs_for_graph = tf.nn.softmax(self.outs)
            self.accuracy = masked_accuracy(tf.nn.softmax(self.outs), self.placeholders['labels'], self.placeholders['labels_mask'])
            self.mo_accuarcy = masked_accuracy(self.outputs_weight*tf.nn.softmax(self.outs)+tf.nn.softmax(self.outputs), self.placeholders['labels'], self.placeholders['labels_mask'])
        else:
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                            self.placeholders['labels_mask'])


    def _accuracy_of_class(self):

       # if self.is_gcn:
            #self.merged = tf.math.add(tf.math.scalar_mul(self.outputs_weight, self.outputs),
                                     # tf.math.scalar_mul(self.outs_weight, self.outs))
           # self.accuracy_of_class = [masked_accuracy(self.merged, self.placeholders['labels'],
                                     # self.placeholders['labels_mask'] * self.placeholders['labels'][:, i])
                                     # for i in range(self.placeholders['labels'].shape[1])]
        #else:
        self.accuracy_of_class = [masked_accuracy(self.outputs, self.placeholders['labels'],
                                      self.placeholders['labels_mask'] * self.placeholders['labels'][:, i])
                                      for i in range(self.placeholders['labels'].shape[1])]

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
