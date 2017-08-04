#' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
#' @param ip: Input keras tensor
#' @param nb_filter: number of filters
#' @param bottleneck: add bottleneck block
#' @param dropout_rate: dropout rate
#' @param weight_decay: weight decay factor
#'
#' @references https://github.com/titu1994/DenseNet/blob/master/densenet.py
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
    kernel_size = c(3,3),
    kernel_initializer = "he_uniform",
    padding = "same",
    kernel_regularizer = keras::regularizer_l2(weight_decay)
  )

  if(!is.null(dropout_rate)){
    x <- keras::layer_dropout(x, rate = dropout_rate)
  }

  x
}


#' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
#'
#' @param ip: Input keras tensor
#' @param nb_filter: number of filters
#' @param compression: calculated as 1 - reduction. Reduces the number of
#'                     feature maps in the transition block.
#' @param dropout_rate: dropout rate
#' @param weight_decay: weight decay factor
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

#' Build a dense_block where the output of each conv_block is fed to subsequent
#' ones
#'
#'
#' @param x: keras tensor
#' @param nb_layers: the number of layers of conv_block to append to the model.
#' @param nb_filter: number of filters
#' @param growth_rate: growth rate
#' @param bottleneck: bottleneck block
#' @param dropout_rate: dropout rate
#' @param weight_decay: weight decay factor
#' @param grow_nb_filters: flag to decide to allow number of filters to grow
#' @param return_concat_list: return the list of feature maps along with the actual output
dense_block <- function(x, nb_layers, nb_filter, growth_rate,
                        bottleneck = FALSE,
                        dropout_rate = NULL,
                        weight_decay = 1e-4,
                        grow_nb_filters = TRUE,
                        return_concat_list = FALSE) {

  x_list <- list(x)

  for (i in 1:nb_layers) {

    cb <- conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
    x_list[[i+1]] <- cb

    x <- layer_concatenate(list(x, cb), axis = concat_axis)

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
#' @param ip: keras tensor
#' @param nb_filters: number of layers
#' @param type: can be 'upsampling', 'subpixel', 'deconv'. Determines type of upsampling performed
#' @param weight_decay: weight decay factor
#'
transition_up_block <- function(ip, nb_filters, type = "upsampling", weight_decay = 1e-4) {

  if (type == "upsampling") {

    x <- keras::layer_upsampling_2d(ip)

  } else if (type = "subpixel") {

    stop("subpixel not implemented")

  } else {

    x <- keras::layer_conv_2d_transpose(
      ip,
      filters = nb_filters,
      kernel_size = c(3,3) ,
      activation = "relu",
      padding = "same",
      kernel_regularizer = keras::regularizer_l2(weight_decay),
      kernel_initializer = "he_uniform"
      )

  }

  x
}

#' Build the DenseNet model
#'
#'
#' @param nb_classes: number of classes
#' @param img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
#' @param include_top: flag to include the final Dense layer
#' @param nb_dense_block: number of dense blocks to add to end (generally = 3)
#' @param growth_rate: number of filters to add per dense block
#' @param reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
#' @param dropout_rate: dropout rate
#' @param weight_decay: weight decay
#' @param nb_layers_per_block: number of layers in each dense block.
#'            Can be a positive integer or a list.
#'            If positive integer, a set number of layers per dense block.
#'            If list, nb_layer is used as provided. Note that list size must
#'            be (nb_dense_block + 1)
#' @param nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
#' @param upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
#'            type of upsampling algorithm used.
#' @param input_shape: Only used for shape inference in fully convolutional networks.
#' @param activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
#'                    Note that if sigmoid is used, classes must be 1.
create_dense_net <- function(nb_classes, img_input, include_top, depth = 40,
                             nb_dense_block = 3, growth_rate = 12, nb_filter = -1,
                             nb_layers_per_block = -1, bottleneck = FALSE, reduction=0.0,
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

    if (nb_layers_per_block == 1) {

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

  for (bloc_idx in 1:nb_dense_block) {

    aux <- dense_block(x,nb_layers[bloc_idx], nb_filter, growth_rate,
                       bottleneck = bottleneck, dropout_rate = dropout_rate,
                       weight_decay = weight_decay)

    x <- transition_block(aux$x, aux$nb_filter, compression = compression,
                          dropout_rate = dropout_rate,
                          weight_decay = weight_decay)


    nb_filter <- trunc(nb_filter * compression)

  }

  # The last dense_block does not have a transition_block
  aux <- dense_block(x, final_nb_layer, nb_filter, groth_rate,
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


#' Build the DenseNet model
#' @param nb_classes: number of classes
#' @param img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
#' @param include_top: flag to include the final Dense layer
#' @param nb_dense_block: number of dense blocks to add to end (generally = 3)
#' @param growth_rate: number of filters to add per dense block
#' @param reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
#' @param dropout_rate: dropout rate
#' @param weight_decay: weight decay
#' @param nb_layers_per_block: number of layers in each dense block.
#'        Can be a positive integer or a list.
#'        If positive integer, a set number of layers per dense block.
#'        If list, nb_layer is used as provided. Note that list size must
#'        be (nb_dense_block + 1)
#' @param nb_upsampling_conv: number of convolutional layers in upsampling via subpixel convolution
#' @param upsampling_type: Can be one of 'upsampling', 'deconv' and 'subpixel'. Defines
#'        type of upsampling algorithm used.
#' @param input_shape: Only used for shape inference in fully convolutional networks.
#' @param activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
#'        Note that if sigmoid is used, classes must be 1.
#'
create_fcn_dense_net <- function(nb_classes, img_input, include_top, depth = 40,
                             nb_dense_block = 3, growth_rate = 12, nb_filter = -1,
                             nb_layers_per_block = -1, bottleneck = FALSE, reduction=0.0,
                             dropout_rate = NULL, weight_decay = 1e-4, activation = "softmax"){


  if (keras::backend()$image_data_format() == "channels_first"){
    concat_axis <- 1
    rows <- input_shape[2]
    cols <- input_shape[3]
  } else {
    concat_axis <- -1
    rows <- input_shape[1]
    cols <- input_shape[2]
  }

  if (reduction != 0) {
    stopifnot(reduction <= 1.0, reduction > 0.0)
  }


  # check if upsampling_conv has minimum number of filters
  # minimum is set to 12, as at least 3 color channels are needed for correct upsampling
  stopifnot(nb_upsampling_conv > 12, nb_upsampling_conv %% 4 == 0)

  if (length(nb_layers_per_block) > 1) {
    stopifnot(length(nb_layers_per_block) == (nb_dense_block + 1))

    bottleneck_nb_layers <- nb_layers_per_block[length(nb_layers_per_block)]
    rev_layers <- rev(nb_layers_per_block)
    nb_layers_per_block <- c(nb_layers_per_block, rev_layers[-1])

  } else {

    bottleneck_nb_layers <- nb_layers_per_block
    nb_layers <- rep(nb_layers_per_block, 2*nb_dense_block + 1)

  }

  # compute compression factor
  compression <- 1.0 - reduction

  # Initial convolution
  x <- img_input %>%
    keras::layer_conv_2d(
      filters = init_conv_filters,
      kernel_size = c(3,3),
      kernel_initializer = "he_uniform",
      padding = "same",
      name = "initial_conv2D",
      use_bias = FALSE,
      kernel_regularizer = keras::regularizer_l2(weight_decay)
    )

  nb_filter <- init_conv_filters
  skip_list <- list()


  for (bloc_idx in 1:nb_dense_block) {

    # Add dense blocks and transition down block
    aux <- dense_block(
      x,
      nb_layers[block_idx],
      nb_filter,
      growth_rate,
      dropout_rate = dropout_rate,
      weight_decay = weight_decay
    )

    # Skip connection
    skip_list[[bloc_idx]] <- aux$x

    # add transition_block
    x <- transition_block(
      aux$x,
      aux$nb_filter,
      compression = compression,
      dropout_rate = dropout_rate,
      weight_decay = weight_decay
      )

    # this is calculated inside transition_down_block
    nb_filter <- trunc(aux$nb_filter * compression)

  }

  # The last dense_block does not have a transition_down_block
  # return the concatenated feature maps without the concatenation of the input

  aux <- dense_block(
    x,
    bottleneck_nb_layers,
    nb_filter,
    growth_rate,
    dropout_rate = dropout_rate,
    weight_decay = weight_decay,
    return_concat_list = TRUE
  )

  skip_list <- rev(skip_list)  # reverse the skip list

  for (bloc_idx in 1:nb_dense_block) {

    n_filters_keep <- growth_rate*nb_layers[nb_dense_block + bloc_idx]

    # upsampling block must upsample only the feature maps (concat_list[1:]),
    # not the concatenation of the input with the feature maps (concat_list[0].
    l <- keras::layer_concatenate(skip_list[-1], axis = concat_axis)
    t <- transition_up_block(l, nb_filters = nb_filters_keep, type = upsampling_type)

    # concatenate the skip connection with the transition block
    x <- keras::layer_concatenate(list(t, skip_list[[bloc_idx]]), axis = concat_axis)

    # Dont allow the feature map size to grow in upsampling dense blocks
    aux <- dense_block(
      x,
      nb_layers[nb_dense_block + block_idx + 1],
      nb_filter = growth_rate,
      growth_rate = growth_rate,
      dropout_rate = dropout_rate,
      weight_decay = weight_decay,
      return_concat_list = TRUE,
      grow_nb_filters = FALSE
    )

  }

  if (include_top) {

    x <- keras::layer_conv_2d(
      aux$x,
      nb_classes,
      kernel_size = c(1,1),
      activation = "linear",
      padding = "same",
      kernel_regularizer = keras::regularizer_l2(weight_decay),
      use_bias = FALSE
    )

    if (keras::backend()$image_data_format() == "channels_first") {

      channel <- input_shape[1]
      row <- input_shape[2]
      col <- input_shape[3]

    } else {

      channel <- input_shape[3]
      row <- input_shape[2]
      col <- input_shape[1]

    }

    x <- x %>%
      keras::layer_reshape(target_shape = c(row*col, nb_classes)) %>%
      keras::layer_activation(activation) %>%
      keras::layer_reshape(c(row, col, nb_classes))

  } else {
    x <- aux$x
  }

  x
}
