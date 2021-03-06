% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/densenet.R
\name{dense_block}
\alias{dense_block}
\title{Build a dense_block}
\usage{
dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck = FALSE,
  dropout_rate = NULL, weight_decay = 1e-04, grow_nb_filters = TRUE,
  return_concat_list = FALSE)
}
\arguments{
\item{x}{keras tensor}

\item{nb_layers}{the number of layers of conv_block to append to the model.}

\item{nb_filter}{number of filters}

\item{growth_rate}{growth rate}

\item{bottleneck}{bottleneck block}

\item{dropout_rate}{dropout rate}

\item{weight_decay}{weight decay factor}

\item{grow_nb_filters}{flag to decide to allow number of filters to grow}

\item{return_concat_list}{return the list of feature maps along with the
actual output}
}
\description{
Build a dense_block where the output of each conv_block is fed to subsequent
ones
}
\seealso{
Other internal: \code{\link{conv_block}},
  \code{\link{create_dense_net}},
  \code{\link{transition_up_block}}
}
