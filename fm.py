import tensorflow as tf
import numpy as np
import sys
import scipy.sparse

def get_batches(x, y, batch_size):
    total_len = x.shape[0]
    for start in range(0, total_len, batch_size):
        end = min(start + batch_size, total_len)
        yield x[start:end], y[start:end]
        
def matmul_wrapper(a, b):
    if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
        return tf.sparse_tensor_dense_matmul(a, b)
    else:
        return tf.matmul(a,b)
    
def power_wrapper(a, p):
    if isinstance(a, tf.SparseTensor):
        return tf.SparseTensor(indices=a.indices, values=tf.pow(a.values, p), dense_shape=a.dense_shape)
    else:
        return tf.pow(a, p)

class FM():
    
    supported_optimizer = {
        'sgd': tf.train.GradientDescentOptimizer(0.01),
        'adam': tf.train.AdamOptimizer(),
        'adagrad': tf.train.AdagradOptimizer(0.01),
        'adadelta': tf.train.AdadeltaOptimizer()
    }
    supported_loss = {
        'mse': tf.losses.mean_squared_error,
        'log_loss': tf.losses.log_loss,
        'hinge_loss': tf.losses.hinge_loss
    }
    
    supported_input_type = ['dense', 'sparse', 'embedding']
    
    def __init__(self, num_features, num_factors=8, input_type='dense', optimizer='sgd', loss='mse', 
                 l2_regularizer=[0.0, 0.01, 0.01]):
            
        if input_type not in FM.supported_input_type:
            raise Exception("Not supported input type {}. Currently support: {}"
                            .format(input_type, ','.join(FM.supported_input_type)))
            
        if input_type == 'embedding' and False == isinstance(num_features, list):
            raise Exception("num_features must be a list when input_type is set to embedding")
        
        self.num_features = num_features
        self.num_factors = num_factors
        self.input_type = input_type
        
        self._init_inputs()
        self._init_training_variables(num_features)
        self._init_model()
        self._init_loss(loss, l2_regularizer)
        self._init_optimizer(optimizer)
        
    def _has_input_type(self, input_type):
        return self.input_type == input_type
    
    def _init_inputs(self):
        with tf.name_scope("inputs"):
            if self._has_input_type('sparse'):
                self.sparse_indices = tf.placeholder(tf.int64, shape=[None, 2], name='sparse_indices')
                self.sparse_data = tf.placeholder(tf.float32, shape=[None], name='sparse_data')
                self.sparse_shape = tf.placeholder(tf.int64, shape=[2], name='sparse_shape')
                self.inputs = tf.SparseTensor(self.sparse_indices, self.sparse_data, self.sparse_shape)
            elif self._has_input_type('embedding'):
                self.inputs = tf.placeholder(tf.int32, shape=[None, None], name="inputs")
            else:
                self.inputs = tf.placeholder(tf.float32, shape=[None, self.num_features], name="inputs")
            self.targets = tf.placeholder(tf.float32, shape=[None], name="targets")
            
    def _init_training_variables(self, num_features):
        with tf.name_scope("training_variables"):
            self.bias = tf.Variable(tf.zeros(1), name="bias")
            if self._has_input_type('embedding'):
                self.weights = tf.Variable(tf.zeros([1, num_features]), name="weights")
                self.vectors = [tf.Variable(tf.truncated_normal([num, self.num_factors], stddev=0.01)
                                           for num in self.num_features)]
            else:
                self.weights = tf.Variable(tf.zeros([1, num_features]), name="weights")
                self.vectors = tf.Variable(tf.truncated_normal([self.num_factors, num_features], stddev=0.01),
                                           name="vectors")
    
    def _init_model(self):
        with tf.name_scope("model"):
            
            if self._has_input_type('embedding'):
                for idx, vec in enumerate(self.vectors):
                    tf.nn.embedding_lookup(vec, tf.slice(self.inputs, [0, idx], []))
            else:
                wx = matmul_wrapper(self.inputs, tf.transpose(self.weights))
                wxsum = tf.reduce_sum(wx, 1)
                
                d = matmul_wrapper(self.inputs, tf.transpose(self.vectors))
                dsqr = matmul_wrapper(power_wrapper(self.inputs, 2), tf.pow(tf.transpose(self.vectors), 2))
                vecsum = tf.multiply(0.5, tf.reduce_sum(tf.subtract(tf.pow(d, 2), dsqr), axis=1))
                
                self.logits = vecsum + wxsum + self.bias

    def _init_loss(self, loss, l2_regularizer):
        loss_function = FM.supported_loss.get(loss, None)
        if loss_function is None:
            raise Exception("Not supported loss function {}. Currently support: {}"
                            .format(loss, ','.join(FM.supported_loss.keys())))
        
        with tf.name_scope("loss"):
            l2_regularization = tf.nn.l2_loss([self.bias]) * l2_regularizer[0] + \
                        tf.nn.l2_loss([self.weights]) * l2_regularizer[1] + \
                        tf.nn.l2_loss([self.vectors]) * l2_regularizer[2]
            
            self.loss = loss_function(self.targets, self.logits)
            self.target_loss = self.loss + l2_regularization

    def _init_optimizer(self, optimizer):
        if type(optimizer) is str:
            opt_function = FM.supported_optimizer.get(optimizer, None)
            if opt_function is None:
                raise Exception("Not supported optimizer {}. Currently support: {}"
                            .format(optimizer, ','.join(FM.supported_optimizer.keys())))
        else:
            opt_function = optimizer
            
        with tf.name_scope("train"):
            self.optimizer = opt_function.minimize(self.target_loss)
        
    def _batch_to_feeddict(self, batch_x, batch_y=None):
        feed_dict = {}
        
        if self._has_input_type("sparse"):
            if False == scipy.sparse.issparse(batch_x):
                raise Exception("Batch Data is not Sparse")
            coo = batch_x.tocoo()
            feed_dict[self.sparse_indices] = np.mat([coo.row, coo.col]).transpose()
            feed_dict[self.sparse_data] = coo.data
            feed_dict[self.sparse_shape] = coo.shape
        else:
            feed_dict[self.inputs] = batch_x
        
        if batch_y is not None:
            feed_dict[self.targets] = batch_y
        return feed_dict
    
    def init_session(self, config=None):
        self.init_all_variable = tf.global_variables_initializer()        
        self.sess = tf.Session(config=config)
        self.sess.run(self.init_all_variable)
        
    def close_session(self):
        self.sess.close()
    
    def fit(self, x, y, epoch=10, batch_size=32, valid_x=None, valid_y=None):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(y, list):
            y = np.array(y)
        
        do_validation = False
        if valid_x is not None and valid_y is not None:
            do_validation = True
        
        for e in range(epoch):
            trained_cnt = 0
            loss_sum = 0
            for batch_x, batch_y in get_batches(x, y, batch_size):
                feed_dict = self._batch_to_feeddict(batch_x, batch_y)
                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
                
                loss_sum += loss * batch_x.shape[0]
                trained_cnt += batch_x.shape[0]
                sys.stdout.write("\rEpoch {} {}/{} \tloss: {:.4f}"
                                 .format(e + 1, trained_cnt, x.shape[0], loss_sum/float(trained_cnt)))
            
            if do_validation:
                feed_dict = self._batch_to_feeddict(valid_x, valid_y)
                loss = self.sess.run(self.loss, feed_dict=feed_dict)
                
                sys.stdout.write("\tvalid_loss: {:.4f}".format(loss))
                
            sys.stdout.write("\n")
            
    def test(self, x, y):
        feed_dict = self._batch_to_feeddict(x, y)
        loss = self.sess.run(self.loss, feed_dict=feed_dict)
        sys.stdout.write("test_loss: {:.4f}".format(loss))
        return loss
    
    def predict(self, x):
        feed_dict = self._batch_to_feeddict(x)
        logits = self.sess.run(self.logits, feed_dict=feed_dict)
        return logits
    
