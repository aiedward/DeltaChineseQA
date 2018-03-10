import tensorflow as tf
import os
import pickle
from model import RNet


initial_learning_rate = 0.5
batch_size = 64
learning_rate_change_freq = 200
save_freq = 500
iterations = 6000

w_emb = './embeddings/word_emb.pickle'
c_emb = './embeddings/char_emb.pickle'
train_dir = "./data/train.tfrecords"
val_dir = "./data/dev.tfrecords"



def get_batch():
    global train_dir, val_dir

    b = [x for x in range(*[40, 401, 40])]

    def r(key, ele):
        return ele.batch(batch_size)
    
    def k(cid, a, a2, a3, a4, a5, a6):
        c_length = tf.reduce_sum(tf.cast(tf.cast(cid, tf.bool), tf.int32))
        return tf.argmax(tf.clip_by_value(b, 0, c_length))

    def Parser():
        def parse(e):
            f = tf.parse_single_example(e,
                features={"context_idxs": tf.FixedLenFeature([], tf.string),
                        "ques_idxs": tf.FixedLenFeature([], tf.string),
                        "context_char_idxs": tf.FixedLenFeature([], tf.string),
                        "ques_char_idxs": tf.FixedLenFeature([], tf.string),
                        "y1": tf.FixedLenFeature([], tf.string),
                        "y2": tf.FixedLenFeature([], tf.string),
                        "id": tf.FixedLenFeature([], tf.int64)})
            ids = []
            ids.append(tf.reshape(tf.decode_raw(f["context_idxs"], tf.int32), [400]))
            ids.append(tf.reshape(tf.decode_raw(f["ques_idxs"], tf.int32), [50]))
            ids.append(tf.reshape(tf.decode_raw(f["context_char_idxs"], tf.int32), [400, 16]))
            ids.append(tf.reshape(tf.decode_raw(f["ques_char_idxs"], tf.int32), [50, 16]))
            ids.append(f["id"])
            ys = []
            ys.append(tf.reshape(tf.decode_raw(f["y1"], tf.float32), [400])) 
            ys.append(tf.reshape(tf.decode_raw(f["y2"], tf.float32), [400])) 
             
            

            return ids[0], ids[1], ids[2], ids[3], ys[0], ys[1], ids[4]
        return parse
    
    train = tf.data.TFRecordDataset(train_dir).map(Parser()).repeat()
    val = tf.data.TFRecordDataset(val_dir).map(Parser()).repeat()
    train = train.apply(tf.contrib.data.group_by_window(k, r, window_size=5 * batch_size))
    val = val.apply(tf.contrib.data.group_by_window(k, r, window_size=5 * batch_size))
    return train, val

def main():
    global w_emb, c_emb, initial_learning_rate
    
    with open(w_emb, 'rb') as handle:
        w_emb = pickle.load(handle)
    with open(c_emb, 'rb') as handle:
        c_emb = pickle.load(handle)
    
    train, val = get_batch()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, 
        train.output_types, train.output_shapes)
    
    
    result = []
    current_best_loss = 20
    model = RNet(iterator, w_emb[1], c_emb[1])
    learning_rate = initial_learning_rate
    print("start training...")
    print("save every " +str(save_freq)+" iterations")
    print("check loss every " +str(learning_rate_change_freq)+" iterations")
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=500)
        train_handle = sess.run(train.make_one_shot_iterator().string_handle())
        val_handle = sess.run(val.make_one_shot_iterator().string_handle())
        sess.run(tf.assign(model.is_train, tf.constant(True, dtype=tf.bool)))
        sess.run(tf.assign(model.lr, tf.constant(learning_rate)))

        for x in range(1, iterations + 1):
            
            
            loss, _ = sess.run([model.loss, model.train_op], feed_dict={handle: train_handle})
            
            
            if x % learning_rate_change_freq == 0:
                sess.run(tf.assign(model.is_train,
                                   tf.constant(False, dtype=tf.bool)))
                val_loss, _ = sess.run([model.loss, model.train_op], feed_dict={
                                      handle: val_handle})
                sess.run(tf.assign(model.is_train,
                                   tf.constant(True, dtype=tf.bool)))
                if val_loss < current_best_loss:
                    current_best_loss = val_loss
                else:
                    print("learning rate changed")
                    learning_rate *= 0.5
                    sess.run(tf.assign(model.lr, 
                    tf.constant(learning_rate)))
                result.append((val_loss,loss))
            
            if x % save_freq == 0:
                
                filename = os.path.join("./model", "model_{}.ckpt".format(x))
                saver.save(sess, filename)

            
                



if __name__ == '__main__':
    main()


