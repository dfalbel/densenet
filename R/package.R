# Main Keras module
keras <- NULL

.onLoad <- function(libname, pkgname) {
  keras <<- keras::implementation()
}
