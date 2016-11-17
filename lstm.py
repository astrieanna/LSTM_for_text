import tensorflow as tf
import numpy as np
import sys

hidden_state_size = 128
input_size = 99
depth = 1

char_to_index = {9: 96,
                 10: 97,
                 11: 98}
index_to_char = {96: 9,
                 97: 10,
                 98: 11}

def get_onehot_index(char):
  o = ord(char)
  if o in char_to_index:
    index = char_to_index[o]
  elif o > (32 + 96):
    print("\n\tBad char: ", o)
    index = ord(" ") - 32
  else:
   index = o - 32
   assert 0 <= index < input_size, "ord is " + str(o)

  return index

def make_onehot(chars):
  if len(chars) == depth:
    output = np.zeros([depth,input_size], dtype=np.float32)
    for i,c in enumerate(chars):
      index = get_onehot_index(c)
      output[i,index] = 1
  else:
    error("Got " + len(chars) + " characters, but expected " + depth)

  return output

def get_onehot_char(index):
  if index in index_to_char:
   return chr(index_to_char[index])

  return chr(index + 32)

 
def unonehot(mat, soft=False, verbose=False):
  best_is =  [-1 for i in range(0,depth)]
  if verbose:
    print(mat)
  if soft:
    for r, row in enumerate(mat):
      best_is[r] = np.random.choice(len(row), 1, p=row)[0]
  else:
    best_vs = [-1 for i in range(0,depth)] 
    for r, row in enumerate(mat):
      for i, v in enumerate(row):
        if v > best_vs[r]:
          best_vs[r] = v
          best_is[r] = i

  return [get_onehot_char(i) for i in best_is]

 
# Returns new_state and output
def create_LSTM_cell(layer, cprev_state, cprev_output, cinput):
  inputandoldoutput = tf.concat(1, [cprev_output, cinput])

  cmemory_weights = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, layer.cgen_memory_weights), layer.cmemory_weights_bias))  
  cupdate_weights = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, layer.cgen_update_weights), layer.cupdate_weights_bias))
  cupdate_vec = tf.tanh(tf.add(tf.matmul(inputandoldoutput, layer.cgen_update_vec), layer.cupdate_vec_bias))
  calmost_output_vec = tf.sigmoid(tf.add(tf.matmul(inputandoldoutput, layer.cgen_output_weights), layer.calmost_output_bias))

  cweighted_state = tf.mul(cprev_state, cmemory_weights)
  cweighted_update = tf.mul(cupdate_vec, cupdate_weights)
  cnewstate = tf.add(cweighted_state, cweighted_update)
  coutput_vec = tf.mul(calmost_output_vec, tf.tanh(cnewstate))

  return cnewstate, coutput_vec 

class Layer:
  def __init__(self, i, cinput):
    self.hss = hidden_state_size
    if i == 2:
      self.is_=hidden_state_size
    else:
      self.is_=input_size
    self.concat_size = self.hss + self.is_
    self.cinput = cinput
    self.vinitial_state = np.zeros([1,self.hss], dtype=np.float32)
    self.vinitial_prev_output = np.zeros([1,self.hss], dtype=np.float32)
    self.c_iprev_state = tf.placeholder(tf.float32, shape=(1, self.hss))
    self.c_iprev_output = tf.placeholder(tf.float32, shape=(1, self.hss))
    self.newstate_output = []

    # Make the matrixes for generating weightings from prev_state/prev_output/input
    self.cgen_memory_weights = tf.Variable(tf.random_normal((self.concat_size,self.hss)))
    self.cgen_update_weights = tf.Variable(tf.random_normal((self.concat_size, self.hss)))
    self.cgen_update_vec = tf.Variable(tf.random_normal((self.concat_size, self.hss)))
    self.cgen_output_weights = tf.Variable(tf.random_normal((self.concat_size, self.hss)))

    # Biases
    self.cmemory_weights_bias = tf.Variable(tf.random_normal((1,self.hss)))
    self.cupdate_weights_bias = tf.Variable(tf.random_normal((1,self.hss)))
    self.cupdate_vec_bias = tf.Variable(tf.random_normal((1,self.hss)))
    self.calmost_output_bias = tf.Variable(tf.random_normal((1,self.hss)))


    if i == 1:
      self.newstate_output.append(create_LSTM_cell(self, self.c_iprev_state, self.c_iprev_output, tf.slice(self.cinput, [0,0], [1, self.is_])))
      for stepi in range(1,depth):
       self.newstate_output.append(create_LSTM_cell(self, *(self.newstate_output[-1]), tf.slice(self.cinput, [stepi,0], [1, self.is_])))
    elif i == 2:
      self.newstate_output.append(create_LSTM_cell(self, self.c_iprev_state, self.c_iprev_output, self.cinput[0][1]))
      for stepi in range(1,depth):
        self.newstate_output.append(create_LSTM_cell(self, *(self.newstate_output[-1]), self.cinput[0][1]))
    else:
      error("Unexpected layer number")
    

class LSTM:
  def __init__(self, checkpoint_filename):
    self.layer1 = Layer(1, tf.placeholder(tf.float32, shape=(depth,input_size)))
    self.layer2 = Layer(2, self.layer1.newstate_output)
 
    # Gotta translate the output back into character probabilities
    cgen_realoutput = tf.Variable(tf.random_normal((hidden_state_size, input_size))) 
    creal_output_bias = tf.Variable(tf.random_normal((1, input_size)))
    self.creal_outputs = []
    for stepi in range(0,depth):
      self.creal_outputs.append(tf.nn.softmax(tf.add(tf.matmul(self.layer2.newstate_output[stepi][1], cgen_realoutput), creal_output_bias)))
    self.creal_output = tf.concat(0, self.creal_outputs)
  
    self.ccorrect = tf.placeholder(tf.float32, shape=(depth,input_size))
    cost = -tf.reduce_mean(tf.reduce_sum(tf.mul(tf.log(self.creal_output + tf.constant(1e-36)),self.ccorrect))) 
    self.train_step = tf.train.AdamOptimizer().minimize(cost)

    self.session = tf.Session()
    initializer = tf.initialize_all_variables()
    self.session.run([initializer])

    self.saver = tf.train.Saver()
    self.saver_filename = "ALEX"
    if checkpoint_filename != "":
      print("Restoring from checkpoint")
      self.saver.restore(self.session, checkpoint_filename)

    self.nodes_no_train = [*(self.layer1.newstate_output[-1]), *(self.layer2.newstate_output[-1]), self.creal_output]
    self.nodes_with_train = [self.train_step] + self.nodes_no_train

  def reset_state(self):
    for layer in [self.layer1, self.layer2]:
      layer.vnewstate = np.copy(layer.vinitial_state)
      layer.voutput = np.copy(layer.vinitial_prev_output)
    
  def run(self, istraining, currchar, nextchar=None):
    if istraining and nextchar != None:
      _, self.layer1.vnewstate, self.layer1.voutput, self.layer2.vnewstate, self.layer2.voutput, vrealoutput = self.session.run(
           self.nodes_with_train,
           feed_dict={
             self.layer1.cinput: currchar,
             self.layer1.c_iprev_state: self.layer1.vnewstate,
             self.layer1.c_iprev_output: self.layer1.voutput,
             self.layer2.c_iprev_state: self.layer2.vnewstate,
             self.layer2.c_iprev_output: self.layer2.voutput,
             self.ccorrect: nextchar})
    elif not istraining:
      self.layer1.vnewstate, self.layer1.voutput, self.layer2.vnewstate, self.layer2.voutput, vrealoutput =  self.session.run(
           self.nodes_no_train,
           feed_dict={
             self.layer1.cinput: currchar,
             self.layer1.c_iprev_state: self.layer1.vnewstate,
             self.layer1.c_iprev_output: self.layer1.voutput,
             self.layer2.c_iprev_state: self.layer2.vnewstate,
             self.layer2.c_iprev_output: self.layer2.voutput})
    else:
      error("invalid input to run")

    return vrealoutput

  def save(self):
    return self.saver.save(self.session, self.saver_filename)


  def generate_a_sentence(self, c, verbose=False):
    target_len = 100
    output = []
    output.append(c)
    for i in range(1,target_len):
      o = self.run(False, make_onehot([output[-1] for _ in range(0,depth)]))
      output.append(unonehot(o, soft=True, verbose=verbose)[0])

    return ''.join(output)

  def train_a_slice(self, currchar, nextchar):
    return self.run(True, currchar, nextchar=nextchar)

def main(filename, checkpoint_filename):
  print("Reading input text from ", filename)

  lstm = LSTM(checkpoint_filename) # create the nn
  lstm.reset_state()
  count = 0
  generation = 0
  with open(filename, 'r') as f:
    text = f.read()
    i = 1
    while i < (len(text) - depth):
      enext_char = make_onehot([text[j] for j in range(i,i+depth)])
      ecurr_char = make_onehot([text[k] for k in range(i-1, i+depth-1)])

      count += depth 
      if count > 100:
        count = 0
        lstm.reset_state()
        generation += 1
        if generation % 10 == 0:
          print(generation, ": ", lstm.generate_a_sentence('A', verbose=False))
        
      vrealoutput = lstm.train_a_slice(ecurr_char, enext_char)  
      i += depth

  # Save what we learned
  print(lstm.save())

if __name__ == '__main__':
  input_filename = sys.argv[1]
  if len(sys.argv) >= 3:
    checkpoint_filename = sys.argv[2]
  else:
    checkpoint_filename = ""
  main(input_filename, checkpoint_filename)

