import tensorflow as tf
import numpy as np
import sys

hidden_state_size = 64
input_size = 99

char_to_index = {9: 96,
                 10: 97,
                 11: 98}
index_to_char = {96: 9,
                 97: 10,
                 98: 11}

def make_onehot(char):
  o = ord(char)
  if o in char_to_index:
    index = char_to_index[o]
  elif o > (32 + 96):
    print("\n\tBad char: ", o)
    index = ord(" ") - 32
  else:
   index = o - 32
   assert 0 <= index < input_size, "ord is " + str(o)

  output = np.zeros([1,input_size], dtype=np.float32)
  output[0,index] = 1
  return output
 
def unonehot(mat):
  best_i = -1
  best_v = -1
  for row in mat:
    for i, v in enumerate(row):
      if v > best_v:
        best_v = v
        best_i = i

  if best_i in index_to_char:
   return index_to_char[best_i]

  return chr(best_i + 32)
 
# Returns new_state and output
def create_LSTM_cell(cprev_state, cprev_output, cinput):
  concat_size = hidden_state_size + input_size

  # Make the matrixes for generating weightings from prev_state/prev_output/input
  cgen_memory_weights = tf.Variable(tf.random_normal((concat_size,hidden_state_size)))
  cgen_update_weights = tf.Variable(tf.random_normal((concat_size, hidden_state_size)))
  cgen_update_vec = tf.Variable(tf.random_normal((concat_size, hidden_state_size)))
  cgen_output_weights = tf.Variable(tf.random_normal((concat_size, hidden_state_size)))

  # Biases
  cmemory_weights_bias = tf.Variable(tf.random_normal((1,hidden_state_size)))
  cupdate_weights_bias = tf.Variable(tf.random_normal((1,hidden_state_size)))
  cupdate_vec_bias = tf.Variable(tf.random_normal((1,hidden_state_size)))
  calmost_output_bias = tf.Variable(tf.random_normal((1,hidden_state_size)))

  inputandoldoutput = tf.concat(1, [cprev_output, cinput])

  cmemory_weights = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, cgen_memory_weights), cmemory_weights_bias))  
  cupdate_weights = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, cgen_update_weights), cupdate_weights_bias))
  cupdate_vec = tf.tanh(tf.add(tf.matmul(inputandoldoutput, cgen_update_vec), cupdate_vec_bias))
  calmost_output_vec = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, cgen_output_weights), calmost_output_bias))

  cweighted_state = tf.mul(cprev_state, cmemory_weights)
  cweighted_update = tf.mul(cupdate_vec, cupdate_weights)
  cnewstate = tf.add(cweighted_state, cweighted_update)

  coutput_vec = tf.mul(calmost_output_vec, tf.tanh(cnewstate))


  return cnewstate, coutput_vec 

def main(filename):
  print("Reading input text from ", filename)

  cprev_state = tf.placeholder(tf.float32, shape=(1, hidden_state_size))
  cprev_output = tf.placeholder(tf.float32, shape=(1, hidden_state_size))
  vinitial_state = np.zeros([1,hidden_state_size], dtype=np.float32)
  vinitial_prev_output = np.zeros([1,hidden_state_size], dtype=np.float32)
  cinput = tf.placeholder(tf.float32, shape=(1,input_size))
  cnew_state, coutput = create_LSTM_cell(cprev_state, cprev_output, cinput)

  # Gotta translate the output back into character probabilities
  cgen_realoutput = tf.Variable(tf.random_normal((hidden_state_size, input_size))) 
  creal_output_bias = tf.Variable(tf.random_normal((1, input_size)))
  creal_output = tf.nn.softmax(tf.add(tf.matmul(coutput, cgen_realoutput), creal_output_bias))

  ccorrect = tf.placeholder(tf.float32, shape=(1,input_size))
  cost = -tf.reduce_mean(tf.reduce_sum(tf.mul(tf.log(creal_output),ccorrect))) 
  train_step = tf.train.AdamOptimizer().minimize(cost)

  session = tf.Session()
  initializer = tf.initialize_all_variables()
  session.run([initializer])

  vnewstate = np.copy(vinitial_state)
  voutput = np.copy(vinitial_prev_output)
  count = 0
  with open(filename, 'r') as f:
    text = f.read()
    prev = text[0]
    for char in text[1:]:
      correct = make_onehot(char)
      encoded = make_onehot(prev)

      count += 1
      if count > 300:
        print()
        count = 0
        vnewstate = np.copy(vinitial_state)
        voutput = np.copy(vinitial_prev_output)

        
      _, vnewstate, voutput, vrealoutput =  session.run([train_step, cnew_state,coutput,creal_output], feed_dict={cinput: encoded, cprev_state: vnewstate, cprev_output: voutput, ccorrect: correct})

      print(unonehot(vrealoutput), end="")
      prev = char

if __name__ == '__main__':
  filename = sys.argv[1]
  main(filename)

