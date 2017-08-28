# densenet

`densenet` implements the [*Densely Connected Convolutional Networks*](https://arxiv.org/abs/1608.06993) for R
usning [`keras`](https://github.com/rstudio/keras). This implementation is based on Somshubra python/keras implementation
available [here](https://github.com/titu1994/DenseNet).

![]("schema.jpg")

## Installation

You can install densenet from github with:

``` r
# install.packages("devtools")
devtools::install_github("dfalbel/densenet")
```

## Example

You can use `densenet` the same way you would use an [*application*](https://rstudio.github.io/keras/reference/index.html#section-applications) 
function from [`keras`](https://github.com/rstudio/keras) (eg. [`application_vgg16`](https://rstudio.github.io/keras/reference/application_vgg.html))

The following lines show how you would define DenseNet-40-12 to classify images for the cifar10 dataset.

``` r
library(keras)
library(densenet)

input_img <- layer_input(shape = c(32, 32, 3))
model <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = 0.2)

opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = TRUE)

model %>% compile(
  optimizer = opt,
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
```

As much of the code to train the model is for preprocressing the images, we are ommiting a lot
of code. You can see the full code in the packages vignette using `vignette("cifar10-DenseNet-40-12")`.
The model takes ~125s per epoch on a high-end GPU (Nvidia GeForce 1080 Ti).
Final accuracy on test set was 0.9351 versus 0.9300 reported on the paper.
