# Clear workspace and memory
rm(list = ls()); gc(reset = TRUE)

setwd('D:\\MY_video_tutorials')

# Build simple Neural Network model on the XOR data.

#loading keras library
library(keras)

vector1<-c(0,0,1,1)
vector2<-c(0,1,0,1)

train_data <- array(c(vector1, vector2), dim = c(4,2))

target_data <- array(c(0,1,1,0), dim = c(4,1))

#defining a keras sequential model
model <- keras_model_sequential()

#defining the model 

model %>% 
  layer_dense(units = 16, input_shape = 2) %>% 
  layer_dropout(rate=0.1)%>%
  layer_activation(activation = 'relu') %>% 
  layer_dense(units = 1) %>% 
  layer_activation(activation = 'sigmoid')

#compiling the defined model with metric = accuracy and optimiser as adam.
model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  #optimizer = optimizer_adam( lr= 0.0001 , decay = 1e-6 ), 
  metrics = c('accuracy')
  
)

# summary model

summary(model)

#fitting the model on the training dataset
model %>% fit(train_data, target_data, epochs = 500, batch_size = 10)

x<- c(1,0,1,1)
y<- c(1,1,0,0)


test_data_x <- array(c(x, y), dim = c(4,2))
test_data_y <-  array(c(0,1,1,1), dim = c(4,1))

#Evaluating model on the cross validation dataset

pred <- model %>% predict(test_data_x, batch_size = 128)
Y_pred <- round(pred)

# Confusion matrix
CM <- table(Y_pred, test_data_y)
CM

# evaluate the model
evals <- model %>% evaluate(test_data_x, test_data_y, batch_size = 10)

accuracy <- evals[2][[1]]* 100
accuracy

