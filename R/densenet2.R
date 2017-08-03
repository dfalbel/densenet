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
dense_block <- function() {

}

