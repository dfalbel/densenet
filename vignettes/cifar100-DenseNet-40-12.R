## ---- eval = FALSE-------------------------------------------------------
#  # Libraries ---------------------------------------------------------------
#  library(keras)
#  library(densenet)
#  
#  # Parameters --------------------------------------------------------------
#  
#  batch_size <- 64
#  epochs <- 300
#  
#  # Data Preparation --------------------------------------------------------
#  
#  # see ?dataset_cifar10 for more info
#  cifar100 <- dataset_cifar100()
#  
#  # Normalisation
#  for(i in 1:3){
#    mea <- mean(cifar100$train$x[,,,i])
#    sds <- sd(cifar100$train$x[,,,i])
#  
#    cifar100$train$x[,,,i] <- (cifar100$train$x[,,,i] - mea) / sds
#    cifar100$test$x[,,,i] <- (cifar100$test$x[,,,i] - mea) / sds
#  }
#  x_train <- cifar100$train$x
#  x_test <- cifar100$test$x
#  
#  y_train <- to_categorical(cifar100$train$y, num_classes = 100)
#  y_test <- to_categorical(cifar100$test$y, num_classes = 100)
#  
#  # Model Definition -------------------------------------------------------
#  
#  input_img <- layer_input(shape = c(32, 32, 3))
#  model <- application_densenet(include_top = TRUE, input_tensor = input_img, dropout_rate = 0.2, classes = 100)
#  
#  opt <- optimizer_sgd(lr = 0.1, momentum = 0.9, nesterov = TRUE)
#  
#  model %>% compile(
#    optimizer = opt,
#    loss = "categorical_crossentropy",
#    metrics = "accuracy"
#  )
#  
#  # Model fitting -----------------------------------------------------------
#  
#  # callbacks for weights and learning rate
#  lr_schedule <- function(epoch) {
#  
#    if(epoch <= 150) {
#      return(0.1)
#    } else if(epoch > 150 & epoch <= 225){
#      return(0.01)
#    } else {
#      return(0.001)
#    }
#  
#  }
#  
#  lr_reducer <- callback_learning_rate_scheduler(lr_schedule)
#  model_checkpoint <- callback_model_checkpoint("inst/weights/cifar100-DenseNet-40-12.hdf5", monitor = "val_acc")
#  
#  history <- model %>% fit(
#    x_train, y_train,
#    batch_size = batch_size,
#    epochs = epochs,
#    validation_data = list(x_test, y_test),
#    callbacks = list(
#      lr_reducer,
#      model_checkpoint
#    )
#  )
#  
#  plot(history)
#  
#  evaluate(model, x_test, y_test)

