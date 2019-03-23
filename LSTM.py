import tensorflow as tf

lstm_hidden_size = 10       #神经元个数
batch_size = 16
num_steps = 10      #训练数据的序列长度

lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden_size)

state = lstm.zero_state(batch_size,tf.float32)

loss = 0.0

#current_input = tf.get_variable("input",[])

for i in range(num_steps):
    if i > 0:
        tf.get_variable_scope().reuse_variables()

        lstm_output, state = lstm(current_input,state)

        final_output = fully_connected(lstm_output)

        loss += calc_loss(final_output,expected_output)