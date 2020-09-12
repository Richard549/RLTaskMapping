import tensorflow as tf
import logging

def masked_huber_loss(preds, labels, mask):

	masked_preds = tf.boolean_mask(preds, mask)
	masked_labs = tf.boolean_mask(labels, mask)

	loss = tf.losses.huber_loss(masked_preds, masked_labs) 
	return loss

def masked_mean_squared_error(preds, labels, mask):

	masked_preds = tf.squeeze(tf.boolean_mask(preds, mask))
	masked_labs = tf.boolean_mask(labels, mask)

	print_op = tf.print("mask shape: ", tf.shape(mask), ", mask: ", mask, ". preds shape: ", tf.shape(preds), ", preds: ", preds, ", masked_preds (", tf.shape(masked_preds), "): ", masked_preds, ". labels shape: ", tf.shape(labels), ", labels: ", labels, ", masked_labs (", tf.shape(masked_labs), "): ", masked_labs, summarize=-1)

	loss = tf.losses.mean_squared_error(masked_preds, masked_labs) 
	#loss = tf.losses.mean_squared_error(preds, labels) 
	return loss, print_op

def masked_softmax_cross_entropy(preds, labels, mask):
	masked_preds = tf.boolean_mask(preds, mask)
	masked_labs = tf.boolean_mask(labels, mask)
	
	# TODO check this makes sense...
	# the logits must be a valid probability distribution...
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=masked_preds, labels=masked_labs)
	return tf.reduce_mean(loss), print_op
