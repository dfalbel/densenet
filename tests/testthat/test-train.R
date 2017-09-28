context("works for a simple model")


test_that("simple model", {

  # Libraries ---------------------------------------------------------------
  library(keras)
  library(densenet)

  # Parameters --------------------------------------------------------------

  batch_size <- 16
  epochs <- 1

  # Data Preparation --------------------------------------------------------

  # see ?dataset_cifar10 for more info
  cifar100 <- dataset_cifar100()

  cifar100$train$x <- cifar100$train$x[1:1000,,,]
  cifar100$train$y <- cifar100$train$y[1:1000,]

  cifar100$test$x <- cifar100$test$x[1:1000,,,]
  cifar100$test$y <- cifar100$test$y[1:1000,]

  # Normalisation
  for(i in 1:3){
    mea <- mean(cifar100$train$x[,,,i])
    sds <- sd(cifar100$train$x[,,,i])

    cifar100$train$x[,,,i] <- (cifar100$train$x[,,,i] - mea) / sds
    cifar100$test$x[,,,i] <- (cifar100$test$x[,,,i] - mea) / sds
  }
  x_train <- cifar100$train$x
  x_test <- cifar100$test$x

  y_train <- to_categorical(cifar100$train$y, num_classes = 100)
  y_test <- to_categorical(cifar100$test$y, num_classes = 100)

  # Model Definition -------------------------------------------------------

  input_img <- layer_input(shape = c(32, 32, 3))
  model <- application_densenet(include_top = TRUE, input_tensor = input_img,
                                dropout_rate = 0.2, depth = 16, classes = 100)

  opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = TRUE)

  model %>% compile(
    optimizer = opt,
    loss = "categorical_crossentropy",
    metrics = "accuracy"
  )

  # Model fitting -----------------------------------------------------------

  # callbacks for weights and learning rate
  lr_schedule <- function(epoch) {

    if(epoch <= 150) {
      return(0.1)
    } else if(epoch > 150 & epoch <= 225){
      return(0.01)
    } else {
      return(0.001)
    }

  }

  lr_reducer <- callback_learning_rate_scheduler(lr_schedule)

  history <- model %>% fit(
    x_train, y_train,
    batch_size = 64,
    epochs = 1L,
    validation_data = list(x_test, y_test),
    callbacks = list(
      lr_reducer
    ),
    verbose = 0
  )

  expect_equal(class(history), "keras_training_history")
})

