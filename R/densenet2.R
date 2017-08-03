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
#'
create_dense_net <- function(nb_classes, img_input, include_top, depth = 40,
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


}
