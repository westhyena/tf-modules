import tensorflow as tf
import numpy as np
import sys
import scipy.sparse

def get_batches(x, y, batch_size):
    total_len = x.shape[0]
    for start in range(0, total_len, batch_size):
        end = min(start + batch_size, total_len)
        yield x[start:end], y[start:end]
        
def matmul(a, b):
    if isinstance(a, tf.SparseTensor) or isinstance(b, tf.SparseTensor):
        return tf.sparse_tensor_dense_matmul(a, b)
    else:
        return tf.matmul(a,b)

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
        'hinge_loss ': tf.losses.hinge_loss
    }
    
    def __init__(self, num_features, num_factors=8, sparse=False, optimizer='sgd', loss='mse', 
                 l2_regularizer=[0.0, 0.01, 0.01]):
        self.num_features = num_features
        self.num_factors = num_factors
        self.sparse = sparse
        
        self._init_inputs()
        self._init_training_variables(num_features)
        self._init_model()
        self._init_loss(loss, l2_regularizer)
        self._init_optimizer(optimizer)
    
    def _init_inputs(self):
        with tf.name_scope("inputs"):
            if self.sparse:
                self.sparse_indices = tf.placeholder(tf.int64, shape=[None, 2], name='sparse_indices')
                self.sparse_data = tf.placeholder(tf.float32, shape=[None], name='sparse_data')
                self.sparse_shape = tf.placeholder(tf.int64, shape=[2], name='sparse_shape')
                self.inputs = tf.SparseTensor(self.sparse_indices, self.sparse_data, self.sparse_shape)
            else:
                self.inputs = tf.placeholder(tf.float32, shape=[None, None], name="inputs")
            self.targets = tf.placeholder(tf.float32, shape=[None], name="targets")
            
    def _init_training_variables(self, num_features):
        with tf.name_scope("training_variables"):
            self.bias = tf.Variable(tf.zeros(1), name="bias")
            self.weights = tf.Variable(tf.zeros([1, num_features]), name="weights")
            self.vectors = tf.Variable(tf.truncated_normal([self.num_factors, num_features], stddev=0.01), name="vectors")
    
    def _init_model(self):
        with tf.name_scope("model"):
            wx = matmul(self.inputs, tf.transpose(self.weights))
            wxsum = tf.reduce_sum(wx, 1)
            vecsum = 0
            for f in range(self.num_factors):
                #d = matmul(self.inputs, tf.reshape(self.vectors[f], shape=(self.vectors[f].shape[0].value, 1)))
                d = tf.multiply(self.inputs, tf.transpose(self.vectors[f]))
                fsum = tf.reduce_sum(d, 1)
                fsum_sqr = tf.reduce_sum(tf.pow(d, 2), 1)

                vecsum += tf.multiply(0.5, tf.add(tf.pow(fsum, 2), fsum_sqr))

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
        
        if self.sparse:
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
    
