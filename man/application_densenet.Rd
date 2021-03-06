% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/densenet.R
\name{application_densenet}
\alias{application_densenet}
\title{Instantiate the DenseNet architecture}
\usage{
application_densenet(input_shape = NULL, depth = 40, nb_dense_block = 3,
  growth_rate = 12, nb_filter = 16, nb_layers_per_block = -1,
  bottleneck = FALSE, reduction = 0, dropout_rate = 0,
  weight_decay = 1e-04, include_top = TRUE, weights = NULL,
  input_tensor = NULL, classes = 10, activation = "softmax")
}
\arguments{
\item{input_shape}{optional shape tuple, only to be specified
if `include_top` is False (otherwise the input shape
has to be `(32, 32, 3)` (with `channels_last` dim ordering)
or `(3, 32, 32)` (with `channels_first` dim ordering).
It should have exactly 3 inputs channels,
and width and height should be no smaller than 8.
E.g. `(200, 200, 3)` would be one valid value.}

\item{depth}{number of layers in the DenseNet}

\item{nb_dense_block}{number of dense blocks to add to end (generally = 3)}

\item{growth_rate}{number of filters to add per dense block}

\item{nb_filter}{initial number of filters. -1 indicates initial
number of filters is 2 * growth_rate}

\item{nb_layers_per_block}{number of layers in each dense block.
Can be a -1, positive integer or a list.
If -1, calculates nb_layer_per_block from the network depth.
If positive integer, a set number of layers per dense block.
If list, nb_layer is used as provided. Note that list size must
be (nb_dense_block + 1)}

\item{bottleneck}{flag to add bottleneck blocks in between dense blocks}

\item{reduction}{reduction factor of transition blocks.
Note : reduction value is inverted to compute compression.}

\item{dropout_rate}{dropout rate}

\item{weight_decay}{weight decay factor}

\item{include_top}{whether to include the fully-connected
layer at the top of the network.}

\item{weights}{one of `None` (random initialization) or
cifar10' (pre-training on CIFAR-10)..}

\item{input_tensor}{optional Keras tensor (i.e. output of `layers.Input()`)
to use as image input for the model.}

\item{classes}{optional number of classes to classify images
into, only to be specified if `include_top` is True, and
if no `weights` argument is specified.}

\item{activation}{Type of activation at the top layer. Can be one of
'softmax' or 'sigmoid'.
Note that if sigmoid is used, classes must be 1.}
}
\description{
Instantiate the DenseNet architecture, optionally loading weights pre-trained
on CIFAR-10. Note that when using TensorFlow,
for best performance you should set
`image_data_format='channels_last'` in your Keras config
at ~/.keras/keras.json.
The model and the weights are compatible with both
TensorFlow and Theano. The dimension ordering
convention used by the model is the one
specified in your Keras config file.
}
