import tensorflow as tf
import tensorflow.contrib.layers as layers


def build_q_func(network, hiddens=[256], dueling=True, layer_norm=False, **network_kwargs):
	if isinstance(network, str):
		from baselines.common.models import get_network_builder
		network = get_network_builder(network)(**network_kwargs)

	def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
		with tf.variable_scope(scope, reuse=reuse):
			inputs = tf.concat([input_placeholder, tf.compat.v1.image.rot90(input_placeholder, 1), tf.compat.v1.image.rot90(input_placeholder, 3)], axis = 0)
			# latent = network(input_placeholder)
			print(inputs.get_shape().as_list())
			latent = network(inputs)
			rotate_labels = [tf.zeros(shape = [tf.shape(input_placeholder)[0]]), tf.ones(shape = [tf.shape(input_placeholder)[0]]), 
				tf.ones(shape = [tf.shape(input_placeholder)[0]])*2]
			rotate_labels = tf.concat(rotate_labels, axis = 0)
			rotate_labels = tf.cast(rotate_labels, dtype = tf.int32)
			if isinstance(latent, tuple):
				if latent[1] is not None:
					raise NotImplementedError("DQN is not compatible with recurrent policies yet")
				latent = latent[0]

			latent = layers.flatten(latent)

			with tf.variable_scope("action_value"):
				action_out = latent
				for hidden in hiddens:
					action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
					if layer_norm:
						action_out = layers.layer_norm(action_out, center=True, scale=True)
					action_out = tf.nn.relu(action_out)
				action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)
				rotation_scores = layers.fully_connected(action_out, num_outputs=3, activation_fn=None)
			if dueling:
				with tf.variable_scope("state_value"):
					state_out = latent
					for hidden in hiddens:
						state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
						if layer_norm:
							state_out = layers.layer_norm(state_out, center=True, scale=True)
						state_out = tf.nn.relu(state_out)
					state_score = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
				action_scores_mean = tf.reduce_mean(action_scores, 1)
				action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
				q_out = state_score + action_scores_centered
			else:
				q_out = action_scores
			q_out = tf.split(q_out, 3, axis = 0)[0]
			return q_out, rotation_scores, rotate_labels

	return q_func_builder
