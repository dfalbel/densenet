#' Instantiate the DenseNet architecture
#'
#' Instantiate the DenseNet architecture, optionally loading weights pre-trained
#' on CIFAR-10. Note that when using TensorFlow,
#' for best performance you should set
#' `image_data_format='channels_last'` in your Keras config
#' at ~/.keras/keras.json.
#' The model and the weights are compatible with both
#' TensorFlow and Theano. The dimension ordering
#' convention used by the model is the one
#' specified in your Keras config file.
#'
#' @param input_shape optional shape tuple, only to be specified
#' if `include_top` is False (otherwise the input shape
#' has to be `(32, 32, 3)` (with `channels_last` dim ordering)
#' or `(3, 32, 32)` (with `channels_first` dim ordering).
#' It should have exactly 3 inputs channels,
#' and width and height should be no smaller than 8.
#' E.g. `(200, 200, 3)` would be one valid value.
#' @param depth number of layers in the DenseNet
#' @param nb_dense_block number of dense blocks to add to end (generally = 3)
#' @param growth_rate number of filters to add per dense block
#' @param nb_filter initial number of filters. -1 indicates initial
#' number of filters is 2 * growth_rate
#' @param nb_layers_per_block number of layers in each dense block.
#' Can be a -1, positive integer or a list.
#' If -1, calculates nb_layer_per_block from the network depth.
#' If positive integer, a set number of layers per dense block.
#' If list, nb_layer is used as provided. Note that list size must
#' be (nb_dense_block + 1)
#' @param bottleneck flag to add bottleneck blocks in between dense blocks
#' @param reduction reduction factor of transition blocks.
#' Note : reduction value is inverted to compute compression.
#' @param dropout_rate dropout rate
#' @param weight_decay weight decay factor
#' @param include_top whether to include the fully-connected
#' layer at the top of the network.
#' @param weights one of `None` (random initialization) or
#' cifar10' (pre-training on CIFAR-10)..
#' @param input_tensor optional Keras tensor (i.e. output of `layers.Input()`)
#' to use as image input for the model.
#' @param classes optional number of classes to classify images
#' into, only to be specified if `include_top` is True, and
#' if no `weights` argument is specified.
#' @param activation Type of activation at the top layer. Can be one of
#' 'softmax' or 'sigmoid'.
#' Note that if sigmoid is used, classes must be 1.
#'
#' @export
application_densenet <- function(input_shape = NULL, depth = 40,
                                 nb_dense_block = 3, growth_rate = 12,
                                 nb_filter = 16, nb_layers_per_block = -1,
                                 bottleneck = FALSE, reduction = 0.0,
                                 dropout_rate = 0.0, weight_decay = 1e-4,
                                 include_top = TRUE, weights = NULL,
                                 input_tensor = NULL, classes = 10,
                                 activation = "softmax"){


  # Determine proper input shape
  input_shape <- obtain_input_shape()(
    input_shape, default_size = 32, min_size = 8,
    data_format = keras::backend()$image_data_format(),
    include_top = include_top
  )

  if (is.null(input_tensor)) {

    img_input <- keras::layer_input(shape = input_shape)

  } else {

    if (!keras::backend()$is_keras_tensor(input_tensor)) {

      img_input <- keras::layer_input(tensor = input_tensor,
                                      shape = input_shape)

    } else {

      img_input <- input_tensor

    }

  }

  x <- create_dense_net(
    classes,
    img_input,
    include_top,
    depth,
    nb_dense_block,
    growth_rate,
    nb_filter,
    nb_layers_per_block,
    bottleneck,
    reduction,
    dropout_rate,
    weight_decay,
    activation
  )

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if (!is.null(input_tensor)) {

    inputs <- get_source_inputs()(input_tensor)

  } else {

    inputs <- img_input

  }

  model <- keras::keras_model(inputs, x)


  if(!is.null(weights)){
    if (weights == "cifar10") {
      stop("weigths not yet implemented")
    }
  }


  model
}


#' Apply BatchNorm, ...
#'
#' ... Relu, 3x3 Conv2D, optional bottleneck block and dropout
#'
#' @param ip Input keras tensor
#' @param nb_filter number of filters
#' @param bottleneck add bottleneck block
#' @param dropout_rate dropout rate
#' @param weight_decay weight decay factor
#'
#' @references https://github.com/titu1994/DenseNet/blob/master/densenet.py
#'
#' @importFrom magrittr %>%
#'
#' @family internal
#'
conv_block <- function(ip, nb_filter,
                       bottleneck = FALSE,
                       dropout_rate = NULL,
                       weight_decay = 1e-4) {


  if (keras::backend()$image_data_format() == "channels_first"){
    concat_axis <- 1
  } else {
    concat_axis <- -1
  }

  x <- ip %>%
    keras::layer_batch_normalization(
      axis = concat_axis,
      gamma_regularizer = keras::regularizer_l2(weight_decay),
      beta_regularizer = keras::regularizer_l2(weight_decay)
    ) %>%
    keras::layer_activation(activation = "relu")

  if (bottleneck) {

    # Obtained from:
    # https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua
    inter_channel <- nb_filter * 4

    x <- x %>%
      keras::layer_conv_2d(
        filters = inter_channel,
        kernel_size = c(1,1),
        kernel_initializer = "he_uniform",
        padding = "same",
        use_bias = FALSE,
        kernel_regularizer = keras::regularizer_l2(weight_decay)
      )

    if (!is.null(dropout_rate)) {
      x <- keras::layer_dropout(object = x, rate = dropout_rate)
    }

    x %>%
      keras::layer_batch_normalization(
        axis = concat_axis,
        gamma_regularizer = keras::regularizer_l2(weight_decay),
        beta_regularizer = keras::regularizer_l2(weight_decay)
      ) %>%
      keras::layer_activation("relu")

  }

  x <- keras::layer_conv_2d(
    object = x,
    filters = nb_filter,
    kernel_size = c(3,3),
    kernel_initializer = "he_uniform",
    padding = "same",
    use_bias = FALSE,
    kernel_regularizer = keras::regularizer_l2(weight_decay)
  )

  if(!is.null(dropout_rate)){
    x <- keras::layer_dropout(x, rate = dropout_rate)
  }

  x
}


#' Apply BatchNorm, ...
#'
#' ... Relu 1x1, Conv2D, optional compression, dropout and
#' Maxpooling2D
#'
#' @param ip Input keras tensor
#' @param nb_filter number of filters
#' @param compression calculated as 1 - reduction. Reduces the number of
#'                     feature maps in the transition block.
#' @param dropout_rate dropout rate
#' @param weight_decay weight decay factor
#'
transition_block <- function(ip, nb_filter,
                             compression = 1,
                             dropout_rate = NULL,
                             weight_decay = 1e-4) {

  if (keras::backend()$image_data_format() == "channels_first") {
    concat_axis <- 1
  } else {
    concat_axis <- -1
  }

  x <- ip %>%
    keras::layer_batch_normalization(
      axis = concat_axis,
      gamma_regularizer = keras::regularizer_l2(weight_decay),
      beta_regularizer = keras::regularizer_l2(weight_decay)
    ) %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d(
      filters = rep(trunc(nb_filter * compression)),
      kernel_size = c(1,1),
      kernel_initializer = "he_uniform",
      padding = "same",
      use_bias = FALSE,
      kernel_regularizer = keras::regularizer_l2(weight_decay)
    )

  if(!is.null(dropout_rate)){
    x <- keras::layer_dropout(x, rate = dropout_rate)
  }


  x %>%
    keras::layer_average_pooling_2d(
      pool_size = c(2,2),
      strides = c(2,2)
    )
}

#' Build a dense_block
#'
#' Build a dense_block where the output of each conv_block is fed to subsequent
#' ones
#'
#'
#' @param x keras tensor
#' @param nb_layers the number of layers of conv_block to append to the model.
#' @param nb_filter number of filters
#' @param growth_rate growth rate
#' @param bottleneck bottleneck block
#' @param dropout_rate dropout rate
#' @param weight_decay weight decay factor
#' @param grow_nb_filters flag to decide to allow number of filters to grow
#' @param return_concat_list return the list of feature maps along with the
#' actual output
#'
#' @family internal
#'
dense_block <- function(x, nb_layers, nb_filter, growth_rate,
                        bottleneck = FALSE,
                        dropout_rate = NULL,
                        weight_decay = 1e-4,
                        grow_nb_filters = TRUE,
                        return_concat_list = FALSE) {

  if (keras::backend()$image_data_format() == "channels_first") {
    concat_axis <- 1
  } else {
    concat_axis <- -1
  }

  x_list <- list(x)

  for (i in 1:nb_layers) {

    cb <- conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
    x_list[[i+1]] <- cb

    x <- keras::layer_concatenate(list(x, cb), axis = concat_axis)

    if (grow_nb_filters) {
      nb_filter <- nb_filter + growth_rate
    }

  }


  if (return_concat_list) {
    return(list(x = x, nb_filter = nb_filter, x_list = x_list))
  } else {
    return(list(x = x, nb_filter = nb_filter))
  }
}

#' SubpixelConvolutional Upscaling (factor = 2)
#'
#' @param ip keras tensor
#' @param nb_filters number of layers
#' @param type can be 'upsampling', 'subpixel', 'deconv'. Determines type of
#' upsampling performed
#' @param weight_decay weight decay factor
#'
#' @family internal
#'
transition_up_block <- function(ip, nb_filters, type = "upsampling",
                                weight_decay = 1e-4) {

  if (type == "upsampling") {

    x <- keras::layer_upsampling_2d(ip)

  } else if (type == "subpixel") {

    stop("subpixel not implemented")

  } else {

    x <- keras::layer_conv_2d_transpose(
      ip,
      filters = nb_filters,
      kernel_size = c(3,3) ,
      strides = c(2,2),
      activation = "relu",
      padding = "same",
      kernel_initializer = "he_uniform"
    )

  }

  x
}

#' Build the DenseNet model
#'
#'
#' @param nb_classes number of classes
#' @param img_input tuple of shape (channels, rows, columns) or (rows, columns,
#'  channels)
#' @param include_top flag to include the final Dense layer
#' @param depth total number of layers
#' @param nb_dense_block number of dense blocks to add to end (generally = 3)
#' @param growth_rate number of filters to add per dense block
#' @param reduction reduction factor of transition blocks. Note : reduction
#' value is inverted to compute compression
#' @param dropout_rate dropout rate
#' @param weight_decay weight decay
#' @param nb_layers_per_block number of layers in each dense block.
#'            Can be a positive integer or a list.
#'            If positive integer, a set number of layers per dense block.
#'            If list, nb_layer is used as provided. Note that list size must
#'            be (nb_dense_block + 1)
#' @param activation Type of activation at the top layer. Can be one of
#' 'softmax' or 'sigmoid'. Note that if sigmoid is used, classes must be 1.
#' @param nb_filter initial number of filters. -1 indicates initial
#' number of filters is 2 * growth_rate
#' @param bottleneck flag to add bottleneck blocks in between dense blocks
#'
#'
#' @family internal
#'
create_dense_net <- function(nb_classes, img_input, include_top, depth = 40,
                             nb_dense_block = 3, growth_rate = 12,
                             nb_filter = -1,
                             nb_layers_per_block = -1, bottleneck = FALSE,
                             reduction=0.0,
                             dropout_rate = NULL, weight_decay=1e-4,
                             activation = "softmax"){


  if (keras::backend()$image_data_format() == "channels_first") {
    concat_axis <- 1
  } else {
    concat_axis <- -1
  }

  stopifnot((depth - 4) %% 3 == 0)

  if (reduction != 0) {
    stopifnot(reduction <= 1.0, reduction > 0.0)
  }

  # layers in each dense block
  if (length(nb_layers_per_block) > 1) {
    stopifnot(length(nb_layers_per_block) == (nb_dense_block + 1))

    final_nb_layer <- nb_layers_per_block[length(nb_layers_per_block)]
    nb_layers <- nb_layers_per_block[-length(nb_layers_per_block)]

  } else {

    if (nb_layers_per_block == -1) {

      count <- trunc((depth - 4) / 3)
      nb_layers <- rep(count, nb_dense_block)
      final_nb_layer <- count

    } else {

      final_nb_layer <- nb_layers_per_block
      nb_layers <- rep(nb_layers_per_block, nb_dense_block)

    }

  }

  if (bottleneck) {
    nb_layers <- trunc(nb_layers/2)
  }

  # compute initial nb_filter if -1, else accept users initial nb_filter
  if (nb_filter <= 0) {
    nb_filter <- 2*growth_rate
  }

  # compute compression factor
  compression <- 1.0 - reduction

  # Initial convolution
  x <- keras::layer_conv_2d(
    img_input,
    nb_filter,
    kernel_size = c(3,3),
    kernel_initializer = "he_uniform", padding = "same",
    name = "initial_conv2D", use_bias = FALSE,
    kernel_regularizer = keras::regularizer_l2(weight_decay)
  )

  for (bloc_idx in 1:(nb_dense_block - 1)) {

    aux <- dense_block(x,nb_layers[bloc_idx], nb_filter, growth_rate,
                       bottleneck = bottleneck, dropout_rate = dropout_rate,
                       weight_decay = weight_decay)

    x <- transition_block(aux$x, aux$nb_filter, compression = compression,
                          dropout_rate = dropout_rate,
                          weight_decay = weight_decay)


    nb_filter <- trunc(aux$nb_filter * compression)

  }

  # The last dense_block does not have a transition_block
  aux <- dense_block(x, final_nb_layer, nb_filter, growth_rate,
                     bottleneck = bottleneck, dropout_rate = dropout_rate,
                     weight_decay = weight_decay)

  x <- keras::layer_batch_normalization(
    aux$x, axis = concat_axis,
    gamma_regularizer = keras::regularizer_l2(weight_decay),
    beta_regularizer = keras::regularizer_l2(weight_decay)
  ) %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_global_average_pooling_2d()


  if (include_top) {

    x <- keras::layer_dense(
      x,
      units = nb_classes,
      activation = activation,
      kernel_regularizer = keras::regularizer_l2(weight_decay),
      bias_regularizer = keras::regularizer_l2(weight_decay)
    )

  }

  x
}



