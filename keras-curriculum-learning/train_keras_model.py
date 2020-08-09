#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import keras
import numpy as np
import keras.backend as K
import time

def compile_model(model, initial_lr=1e-3, loss='categorical_crossentropy',
                  optimizer='sgd', metrics=['accuracy'], momentum=0.0):

    optimizer = keras.optimizers.SGD(initial_lr, momentum=momentum)
    
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics)


def basic_data_function(x_train, y_train, batch, history, model):
    return x_train, y_train

def basic_lr_scheduler(initial_lr, batch, history):
    return initial_lr


def generate_random_batch(x, y, batch_size):
    size_data = x.shape[0]
    cur_batch_idxs = np.random.choice(size_data, batch_size, replace=False)
    return x[cur_batch_idxs, :, :, :], y[cur_batch_idxs,:]


def train_model_batches(model, dataset, num_batches, batch_size=100,
                        test_each=50, batch_generator=generate_random_batch, initial_lr=1e-3,
                        lr_scheduler=basic_lr_scheduler, loss='categorical_crossentropy',
                        data_function=basic_data_function,
                        verbose=False):
    
    x_train = dataset.x_train
    y_train = dataset.y_train_labels
    x_test = dataset.x_test
    y_test = dataset.y_test_labels

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "batch_num": [], "data_size": []}
    start_time = time.time()
    for batch in range(num_batches):
        cur_x, cur_y = data_function(x_train, y_train, batch, history, model)
        cur_lr = lr_scheduler(initial_lr, batch, history)
        K.set_value(model.optimizer.lr, cur_lr)
        batch_x, batch_y = batch_generator(cur_x, cur_y, batch_size)
        cur_loss, cur_accuracy = model.train_on_batch(batch_x, batch_y)
        history["loss"].append(cur_loss)
        history["acc"].append(cur_accuracy)
        history["data_size"].append(cur_x.shape[0])
        if test_each is not None and (batch+1) % test_each == 0:
            cur_val_loss, cur_val_acc = model.evaluate(x_test, y_test, verbose=0)
            history["val_loss"].append(cur_val_loss)
            history["val_acc"].append(cur_val_acc)
            history["batch_num"].append(batch)
            if verbose:
                print("Validation Accuracy:", cur_val_acc)
        if verbose and (batch+1) % 5 == 0:
            print("Batch Number: " + str(batch+1) + r"/" + str(num_batches))
            print("Current Learning Rate: " + str(cur_lr))
            print("Loss: " + str(cur_loss))
            print("--------------------")
            start_time = time.time()

    return history