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


