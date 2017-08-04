# Main Keras module
keras_dense <- NULL

.onLoad <- function(libname, pkgname) {
  keras_dense <<- keras::implementation()
}

obtain_input_shape <- function(){
  keras_dense$applications$imagenet_utils$`_obtain_input_shape`
}

get_source_inputs <- function(){
  keras_dense$engine$topology$get_source_inputs
}
