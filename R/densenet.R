

# input_shape=None,
# depth=40,
# nb_dense_block=3,
# growth_rate=12,
# nb_filter=16,
# nb_layers_per_block=-1,
# bottleneck=False,
# reduction=0.0,
# dropout_rate=0.0,
# weight_decay=1E-4,
# include_top=True,
# weights='cifar10',
# input_tensor=None,
# classes=9,
# activation='sigmoid'

obtain_input_shape <- function(...) {
  keras$applications$imagenet_utils$`_obtain_input_shape`(...)
}



applications_densenet <- function(input_shape = NULL, include_top = TRUE, input_tensor = NULL){

  # some checks on the input
  stopifnot(weights == "cifar10" | is.null(weights))
  stopifnot((weights == "cifar10" & classes == 10 & include_top) |
              weights != "cifar10")
  stopifnot(activation %in% c("sigmoid", "softmax"))
  stopifnot((activation == "sigmoid" & classes == 1) |
              activation != "sigmoid")


  input_shape <- obtain_input_shape(
    input_shape,
    default_size = 32,
    min_size = 8,
    data_format = keras::backend()$image_data_format(),
    include_top = include_top
  )

  if(is.null(input_tensor)) {
    img_input <- keras::layer_input(shape = input_shape)
  } else if (!keras::backend()$is_keras_tensor(input_tensor)) {
    img_input <- keras::layer_input(tensor = input_tensor, shape = input_shape)
  } else {
    img_input <- input_tensor
  }

}

create_densenet <- function(nb_classes, img_input, include_top, depth = 40,
                            nb_dense_block = 3, growth_rate = 12, nb_filter = -1,
                            nb_layers_per_block = -1, bottleneck = FALSE, reduction = 0.0,
                            dropout_rate = NULL, weight_decay = 1E-4,
                            activation = "softmax"){

  concat_axis <- ifelse(keras::backend()$image_data_format() == "channels_first", 1, -1)

  stopifnot((depth - 4) %% 3 == 0)
  if(reduction != 0){
    stopifnot(reduction > 0 & reduction <= 1)
  }

  if(length(nb_layers_per_block) > 1){
    nb_layers <- as.list(nb_layers_per_block)
    stopifnot(length(nb_layers) == (nb_dense_block + 1))

    final_nb_layer <- as.list(nb_layers[[length(nb_layers)]])
    nb_layers <- as.list(nb_layers[[-length(nb_layers)]])

  } else if (nb_layers_per_block == -1){

    count <- trunc((depth - 4) / 3)
    nb_layers <- as.list(rep(count, nb_dense_block))
    final_nb_layer <- as.list(count)

  } else {

    final_nb_layer <- as.list(nb_layers_per_block)
    nb_layers <- as.list(nb_layers_per_block * nb_dense_block)

  }

  if(bottleneck){
    nb_layers <- lapply(nb_layers, function(x) trunc(x / 2))
  }

  if(nb_filter <= 0){
    nb_filter <- 2*growth_rate
  }

  compression <- 1 - reduction

  # Initial convolution
  x <- keras::layer_conv_2d(
    img_input,
    filters = nb_filter,
    kernel_size = list(3,3),
    kernel_initializer = "he_uniform",
    padding = "same",
    name = "initial_conv2D",
    use_bias = FALSE,
    kernel_regularizer = keras::regularizer_l2(weight_decay)
    )


  for(block_idx in 1:nb_dense_block) {



  }


}






