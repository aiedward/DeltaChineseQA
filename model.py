import tensorflow as tf
from tensorflow.python.ops.rnn_cell import GRUCell

batch_size = 64
char_limit = 16
dropout_rate = 0.5
h_size = 75
char_dim = 8
char_hidden = 100

class gru:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.params = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            
            if layer == 0:
                input_size_ = input_size
            else:
                input_size_ = num_units * 2
            self.grus.append((tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_), tf.contrib.cudnn_rnn.CudnnGRU(
                num_layers=1, num_units=num_units, input_size=input_size_), ))
            self.params.append((tf.Variable(tf.random_uniform(
                [self.grus[-1][0].params_size()], -0.1, 0.1), validate_shape=False), tf.Variable(tf.random_uniform(
                [self.grus[-1][1].params_size()], -0.1, 0.1), validate_shape=False), ))
            self.inits.append((tf.Variable(tf.zeros([1, batch_size, num_units])), tf.Variable(tf.zeros([1, batch_size, num_units])), ))
            self.dropout_mask.append((dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None), dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode=None), ))
            
    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        
        for layer in range(self.num_layers):
            lul = []
            with tf.variable_scope("fw"):
                out_fw, _ = self.grus[layer][0](outputs[-1] * self.dropout_mask[layer][0], self.inits[layer][0], self.params[layer][0])
                lul.append(out_fw)
            with tf.variable_scope("bw"):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * self.dropout_mask[layer][1], seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = self.grus[layer][1](inputs_bw, self.inits[layer][1], self.params[layer][1])
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                lul.append(out_bw)
            outputs.append(tf.concat(lul, axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res

class ptr_net:
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="ptr_net"):
        self.gru = GRUCell(hidden)
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = dropout(tf.ones(
            [self.batch, hidden], dtype=tf.float32), keep_prob=self.keep_prob, is_train=is_train)

    def __call__(self, init, match, d, mask):
        with tf.variable_scope(self.scope):
            result = []
            inp, l1 = pointer(dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train), init * self.dropout_mask, d, mask)
            result.append(l1)
            d_inp = dropout(inp, keep_prob=self.keep_prob,
                            is_train=self.is_train)
            _, state = self.gru(d_inp, init)
            tf.get_variable_scope().reuse_variables()
            _, l2 = pointer(dropout(match, keep_prob=self.keep_prob,
                              is_train=self.is_train), state * self.dropout_mask, d, mask)
            result.append(l2)
            return result[0], result[1]


def dropout(args, keep_prob, is_train, mode="recurrent"):
    if keep_prob < 1.0:
        if mode == "embedding":
             
            noise_shape = [tf.shape(args)[0], 1]
            args = tf.cond(is_train, lambda: tf.nn.dropout(
                args, keep_prob, noise_shape=noise_shape) * keep_prob, lambda: args)
        if mode == "recurrent" and len(args.get_shape().as_list()) == 3:
            
            noise_shape = [tf.shape(args)[0], 1, tf.shape(args)[-1]]
            args = tf.cond(is_train, lambda: tf.nn.dropout(
                args, keep_prob, noise_shape=noise_shape) * 1.0, lambda: args)
    
    return args

def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope): 
        s = -1e30 * (1 - tf.cast(mask, tf.float32)) + tf.squeeze(affine(tf.nn.tanh(affine(tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2), hidden, use_bias=False, scope="s0")), 1, use_bias=False, scope="s"), [2])
        res = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(s), axis=2) * inputs, axis=1)
        return res, s

def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):  
        s = -1e30 * (1 - tf.cast(mask, tf.float32)) + tf.squeeze(affine(tf.nn.tanh(affine(dropout(memory, keep_prob=keep_prob, is_train=is_train), hidden, scope="s0")), 1, use_bias=False, scope="s"), [2]) 
        res = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(s), axis=2) * memory, axis=1)
        return res


def attention(inputs, memory, mask, hidden, keep_prob=1.0, is_train=None, scope="dot_attention"):
    with tf.variable_scope(scope):

        with tf.variable_scope("attention"):
            
            outputs = tf.matmul(tf.nn.relu(
                affine(dropout(inputs, keep_prob=keep_prob, is_train=is_train), hidden, use_bias=False, scope="inputs")), tf.transpose(
                tf.nn.relu(
                affine(dropout(memory, keep_prob=keep_prob, is_train=is_train), hidden, use_bias=False, scope="memory")), [0, 2, 1])) / (hidden ** 0.5)
            
            logits = tf.nn.softmax(-1e30 * (1 - tf.cast(tf.tile(tf.expand_dims(mask, axis=1), [1, tf.shape(inputs)[1], 1]), tf.float32)) + outputs)
            res = tf.concat([inputs, tf.matmul(logits, memory)], axis=2)

        with tf.variable_scope("gate"):  
            gate = tf.nn.sigmoid(affine(dropout(res, keep_prob=keep_prob, is_train=is_train), res.get_shape().as_list()[-1], use_bias=False))
            return res * gate


def affine(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        if use_bias:
            b = tf.get_variable("b", [hidden], initializer=tf.constant_initializer(0.))

        flat_inputs = tf.reshape(inputs, [-1, inputs.get_shape().as_list()[-1]])
        W = tf.get_variable("W", [inputs.get_shape().as_list()[-1], hidden])
        res = tf.matmul(flat_inputs, W)
        
        if use_bias:
            res = tf.nn.bias_add(res, b)
        
        out_shape = [tf.shape(inputs)[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden]
        res = tf.reshape(res, out_shape)
        return res


class RNet(object):
    
    global batch_size, char_limit, h_size, char_dim, char_hidden
    
    def __init__(self, data, wm, cm, trainable=True):
        
        
        self.set_variable(data, wm, cm)
        c,q = self.emb()
        c,q = self.encodeing(c,q)
        att = self.att(c,q)
        match = self.match(att)
        logits1, logits2 = self.pointer(match,q)
        self.pred(logits1,logits2)

        if trainable == True:
            self.make_train()
        else:
            print("******** Testing Version ********")
            

    def emb(self): 
        with tf.variable_scope("emb"):
            with tf.variable_scope("char"):
                c_context_embs = []
                c_question_embs = []
                c_context_embs.append(tf.reshape(tf.nn.embedding_lookup(
                    self.char_embedding, self.context_hidden), [batch_size * tf.reduce_max(self.context_length), char_limit, char_dim]))
                c_question_embs.append(tf.reshape(tf.nn.embedding_lookup(
                    self.char_embedding, self.question_hidden), [batch_size * tf.reduce_max(self.question_length), char_limit, char_dim]))
                 
                c_context_embs.append(dropout(
                    c_context_embs[0], keep_prob=dropout_rate, is_train=self.is_train))
                c_question_embs.append(dropout(
                    c_question_embs[0], keep_prob=dropout_rate, is_train=self.is_train))
                 
                cell_fw = tf.contrib.rnn.GRUCell(char_hidden)
                cell_bw = tf.contrib.rnn.GRUCell(char_hidden)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, c_context_embs[1], self.ch_len, dtype=tf.float32)
                c_context_embs.append(tf.concat([state_fw, state_bw], axis=1))
                 
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, c_question_embs[1], self.question_hidden_len, dtype=tf.float32)
                question_hidden_emb = tf.concat([state_fw, state_bw], axis=1)
                question_hidden_emb = tf.reshape(question_hidden_emb, [batch_size, tf.reduce_max(self.question_length), 2 * char_hidden])
                context_hidden_emb = tf.reshape(c_context_embs[2], [batch_size, tf.reduce_max(self.context_length), 2 * char_hidden])
            #print("word embedding done!")
            with tf.name_scope("word"):
                context_emb = tf.nn.embedding_lookup(self.word_embedding, self.context)
                question_emb = tf.nn.embedding_lookup(self.word_embedding, self.question)
            #print("char embedding done!")
            context_emb = tf.concat([context_emb, context_hidden_emb], axis=2)
            question_emb = tf.concat([question_emb, question_hidden_emb], axis=2) 
        #print("all embedding done!")
        return context_emb, question_emb 
    
    def encodeing(self,context_emb,question_emb): 
        
        with tf.variable_scope("encoding"):
            layers = []
            layers.append(gru(num_layers=3, num_units=h_size, batch_size=batch_size, input_size=context_emb.get_shape(
            ).as_list()[-1], keep_prob=dropout_rate, is_train=self.is_train))
            layers.append(layers[0](context_emb, seq_len=self.context_length))
            layers.append(layers[0](question_emb, seq_len=self.question_length))
        #print("encoding done!")
        self.encodeing_layers = layers
        return layers[1], layers[2]

    def att(self,c,q):
        with tf.variable_scope("attention"):
            layers = []
            layers.append(attention(c, q, mask=self.question_mask, hidden=h_size,
                                   keep_prob=dropout_rate, is_train=self.is_train)) 
            layers.append(gru(num_layers=1, num_units=h_size, batch_size=batch_size, input_size=layers[0].get_shape(
            ).as_list()[-1], keep_prob=dropout_rate, is_train=self.is_train)) 
            layers.append(layers[1](layers[0], seq_len=self.context_length))
        #print("att done!")
        self.att_layers = layers
        return layers[2]

    def match(self,att):
        with tf.variable_scope("match"):
            layers = []
            layers.append(attention(
                att, att, mask=self.context_mask, hidden=h_size, keep_prob=dropout_rate, is_train=self.is_train))
            
            layers.append(gru(num_layers=1, num_units=h_size, batch_size=batch_size, input_size=layers[0].get_shape(
            ).as_list()[-1], keep_prob=dropout_rate, is_train=self.is_train))
            layers.append(layers[1](layers[0], seq_len=self.context_length))
        self.match_layer = layers
        #print("match done!")
        return layers[2]

    def pointer(self,match,q):
        with tf.variable_scope("pointer"):
            layers = []
            layers.append(summ(q[:, :, -2 * h_size:], h_size, mask=self.question_mask,
                        keep_prob=dropout_rate, is_train=self.is_train))
            layers.append(ptr_net(batch=batch_size, hidden=layers[0].get_shape().as_list(
            )[-1], keep_prob=dropout_rate, is_train=self.is_train))
            
            logits1, logits2 = layers[1](layers[0], match, h_size, self.context_mask)
        #print("pointer done!")
        self.pointer_layer = layers
        return logits1, logits2

    def pred(self,l1,l2):
        with tf.variable_scope("predict"):
            
            self.ys = []
            self.ys.append(tf.argmax(tf.reduce_max(tf.matrix_band_part(tf.matmul(tf.expand_dims(tf.nn.softmax(l1), axis=2),
                              tf.expand_dims(tf.nn.softmax(l2), axis=1)), 0, 15), axis=2), axis=1))
            self.ys.append(tf.argmax(tf.reduce_max(tf.matrix_band_part(tf.matmul(tf.expand_dims(tf.nn.softmax(l1), axis=2),
                              tf.expand_dims(tf.nn.softmax(l2), axis=1)), 0, 15), axis=1), axis=1))
            '''
            self.yp1 = y[0]
            self.yp2 = y[1]
            '''
            loss = []
            loss.append(tf.nn.softmax_cross_entropy_with_logits(
                logits=l1, labels=self.x))
            loss.append(tf.nn.softmax_cross_entropy_with_logits(
                logits=l2, labels=self.y))
            
            loss = tf.add(loss[0],loss[1])
            self.loss = tf.reduce_mean(loss)
        #print("pred done!")
        
    def make_train(self):
        self.lr = tf.get_variable(
                "lr", shape=[], dtype=tf.float32, trainable=False)
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(
                gradients, 5)
        self.train_op = self.opt.apply_gradients(
                zip(capped_grads, variables), global_step=self.steps)
        print("******** Training Version ********")
    
    def set_variable(self,data,wm,cm):
        two_zeros = [0,0]
        three_zeros = [0,0,0]
        self.steps = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)
        context, question, context_hidden, question_hidden, x, y, self.qa_id = data.get_next()
        

        self.context_length = tf.reduce_sum(tf.cast(tf.cast(context, dtype = tf.bool), dtype = tf.int32), axis=1)
        self.question_length = tf.reduce_sum(tf.cast(tf.cast(question, dtype =tf.bool), dtype = tf.int32), axis=1)

        self.context = tf.slice(context, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.question = tf.slice(question, two_zeros, [batch_size, tf.reduce_max(self.question_length)])

        self.context_mask = tf.slice(tf.cast(context, dtype = tf.bool), two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.question_mask = tf.slice(tf.cast(question, dtype =tf.bool), two_zeros, [batch_size, tf.reduce_max(self.question_length)])

        self.context_hidden = tf.slice(context_hidden, three_zeros, [batch_size, tf.reduce_max(self.context_length), char_limit])
        self.question_hidden = tf.slice(question_hidden,three_zeros, [batch_size, tf.reduce_max(self.question_length), char_limit])
        
        self.x = tf.slice(x, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        self.y = tf.slice(y, two_zeros, [batch_size, tf.reduce_max(self.context_length)])
        

        self.ch_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.context_hidden, tf.bool), tf.int32), axis=2), [-1])
        self.question_hidden_len = tf.reshape(tf.reduce_sum(
            tf.cast(tf.cast(self.question_hidden, tf.bool), tf.int32), axis=2), [-1])

        self.is_train = tf.get_variable(
            "is_train", shape=[], dtype=tf.bool, trainable=False)
        self.word_embedding = tf.get_variable("word_mat", initializer=tf.constant(
            wm, dtype=tf.float32), trainable=False)
        self.char_embedding = tf.get_variable(
            "char_mat", initializer=tf.constant(cm, dtype=tf.float32))

    
    def get_loss(self):
        return self.loss

    

