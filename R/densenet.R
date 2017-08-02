#' Apply BtachNorm, Relu, 3x3Conv2D, optional dropout
#'
#' @param x Input keras network
#' @param nb_filter int -- number of filters
#' @param dropout_rate double -- dropout rate
#' @param weight_decay double -- weight decay factor
#'
#' @return keras network with b_norm, relu and convolution2d added
#'
conv_factory <- function(x,
                         nb_filter,
                         dropout_rate = NULL,
                         weight_decay = 1E-4) {

  x <- x %>%
    keras::layer_batch_normalization(
      mode = 0,
      axis = 1,
      gamma_regularizer = keras::regularizer_l2(weight_decay),
      beta_regularizer = keras::regularizer_l2(weight_decay)
    ) %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d(
      filters = nb_filter,
      kernel_size = c(3,3),
      strides = c(3,3),
      padding = "same",
      kernel_initializer = "he_uniform",
      use_bias = FALSE,
      kernel_regularizer = keras::regularizer_l2(weight_decay)
    )

  if(!is.null(dropout_rate)){
    x <- x %>%
      keras::layer_dropout(rate = dropout_rate)
  }

  x
}

#' Apply BatchNorm, Relu, 1x1Conv2D, optional dropout and Maxpooling2D
#'
#' @param x Input keras network
#' @param nb_filter int -- number of filters
#' @param dropout_rate double -- dropout rate
#' @param weight_decay double -- weight decay factor
#'
#' @return keras network with b_norm, relu, convolution2d and maxpooling added
#'
transition <- function(x, nb_filter, dropout_rate = NULL, weight_decay = 1e-4) {

  x <-  x %>%
    keras::layer_batch_normalization(
      mode = 0,
      axis = 1,
      gamma_regularizer = keras::regularizer_l2(weight_decay),
      beta_regularizer = keras::regularizer_l2(weight_decay)
    ) %>%
    keras::layer_activation(activation = "relu") %>%
    keras::layer_conv_2d(
      filters = nb_filter,
      kernel_size = c(1,1),
      strides = c(1,1),
      padding = "same",
      kernel_initializer = "he_uniform",
      use_bias = FALSE,
      kernel_regularizer = keras::regularizer_l2(weight_decay)
    )

  if(!is.null(dropout_rate)){
    x <- x %>%
      keras::layer_dropout(rate = dropout_rate)
  }

  x %>%
    keras::layer_average_pooling_2d(
      pool_size = c(2,2),
      strides = c(2,2)
    )
}
