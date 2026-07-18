# This module is the shared optimization and evaluation layer of the whole
# GTSRB assignment.
# It sits below the top-level experiment controller in main.py and works with
# the model architectures from models.py and the DataLoaders from
# data_pipeline.py.
#
# In the project flow, this module performs:
#
#     model + train_loader + loss + optimizer
#         -> repeated epoch/batch training updates
#         -> list of average epoch losses
#
#     trained model + test_loader
#         -> forward passes on test data
#         -> predicted classes
#         -> classification accuracy
#
# A key design idea in this module is shared procedure:
# - the MLP and CNN should not have separate training code,
# - the MLP and CNN should not have separate evaluation code,
# - both models should be trained and evaluated under the same logic.
#
# That shared procedure is what makes the final comparison in main.py fair and
# easy to interpret. This file therefore acts as the common execution layer for
# both compared architectures.

"""
Shared training and evaluation utilities for the GTSRB assignment.

Current implementation stage:
- REAL training loop is implemented
- REAL evaluation loop is implemented
- both functions can be used for either MLP or CNN

This module is responsible only for:
- training one model
- evaluating one model

It should NOT contain:
- dataset creation
- model architecture definitions
- top-level experiment orchestration

Why this module exists:
The project compares two architectures on the same classification task.
To keep that comparison fair, both models should go through the same:
- training loop structure,
- optimizer-step sequence,
- loss computation logic,
- evaluation procedure.

By placing those shared procedures here, the project avoids duplicated logic
and keeps main.py focused on global experiment control rather than low-level
batch processing.
"""

# torch is needed here for:
# - device transfers,
# - disabling gradients during evaluation,
# - general tensor operations used in both loops.
import torch

# nn is used for type annotations so the training and evaluation functions can
# clearly express that they operate on generic PyTorch modules.
import torch.nn as nn

# DataLoader is used for type annotations to make it explicit that both
# functions consume batched dataset iterators rather than raw datasets.
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> list[float]:
    """
    Train one model for the given number of epochs.

    Parameters:
    - model:
        the neural network to train
    - train_loader:
        DataLoader providing training batches
    - criterion:
        loss function used to compare predictions with true labels
    - optimizer:
        optimizer responsible for updating model parameters
    - device:
        CPU or GPU device on which training should run
    - epochs:
        number of full passes through the training set

    High-level training flow per epoch:
    1. move the model to the selected device,
    2. switch the model into training mode,
    3. iterate over all training batches,
    4. move batch tensors to the selected device,
    5. clear old gradients,
    6. run the forward pass,
    7. compute the loss,
    8. run backpropagation,
    9. update parameters with the optimizer,
    10. accumulate and report average epoch loss.

    Return:
    - list of average training losses, one value per epoch

    Why the returned loss list is useful:
    main.py later uses it to report the final training loss for each model and
    compare how both training runs ended.
    """

    # Ensure the model parameters and all later forward passes operate on the
    # selected device before the training loop starts.
    model.to(device)

    # This list stores one average-loss value per completed epoch.
    # It becomes the compact training summary returned to main.py.
    epoch_losses: list[float] = []

    # -----------------------------------------------------------------
    # Epoch loop
    # -----------------------------------------------------------------
    #
    # One epoch means one full pass through the training DataLoader.
    for epoch in range(epochs):
        # Switch the model into training mode.
        # This matters for layers that behave differently during training and
        # evaluation, even though the current baseline models are simple.
        model.train()

        # These accumulators are used to compute the average loss over the whole
        # epoch in a sample-weighted way.
        running_loss = 0.0
        total_samples = 0

        # -------------------------------------------------------------
        # Batch loop
        # -------------------------------------------------------------
        #
        # Each iteration processes one mini-batch of training images and labels.
        for images, labels in train_loader:
            # Move the current batch to the selected device so the data lives on
            # the same device as the model.
            images = images.to(device)
            labels = labels.to(device)

            # Clear gradients from the previous optimization step.
            # PyTorch accumulates gradients by default, so this reset must
            # happen before each new backward pass.
            optimizer.zero_grad()

            # Step 1:
            # compute the model outputs for the current image batch.
            outputs = model(images)

            # Step 2:
            # compute the loss comparing predicted logits with the true labels.
            loss = criterion(outputs, labels)

            # Step 3:
            # backpropagate through the network to compute gradients of all
            # trainable parameters.
            loss.backward()

            # Step 4:
            # update the model parameters using the optimizer and the newly
            # computed gradients.
            optimizer.step()

            # ---------------------------------------------------------
            # Running loss accumulation
            # ---------------------------------------------------------
            #
            # loss.item() is the average loss over the current batch, so it is
            # multiplied by the batch size to recover the total contribution of
            # that batch. This makes the final epoch average correct even if the
            # last batch is smaller than the others.
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        # Compute the sample-weighted average loss for the whole epoch.
        average_epoch_loss = running_loss / total_samples
        epoch_losses.append(average_epoch_loss)

        # Print one progress line after each epoch so the training run remains
        # observable from the console.
        print(f"Epoch {epoch + 1}/{epochs} - average training loss: {average_epoch_loss:.4f}")

    # Return the full epoch-loss history so the caller can inspect the final
    # training state or compare multiple model runs.
    return epoch_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Evaluate one model on the test set and return classification accuracy.

    Parameters:
    - model:
        the already trained model
    - test_loader:
        DataLoader providing test batches
    - device:
        CPU or GPU device on which evaluation should run

    High-level evaluation flow:
    1. move the model to the selected device,
    2. switch the model into evaluation mode,
    3. disable gradient computation,
    4. iterate over the test batches,
    5. run forward passes,
    6. convert logits to predicted classes with argmax,
    7. count correct predictions,
    8. compute final accuracy in percent.

    Return:
    - accuracy in percent, for example 84.37

    Why this function is separate from training:
    evaluation should reuse the same model interface but should not perform any
    gradient computation or parameter updates. Keeping it separate makes that
    difference explicit.
    """

    # Ensure the model is on the same device as the incoming test batches.
    model.to(device)

    # Switch the model into evaluation mode.
    # This disables training-specific layer behavior where relevant.
    model.eval()

    # These counters accumulate the total number of correct predictions and the
    # total number of evaluated samples across the whole test set.
    correct_predictions = 0
    total_samples = 0

    # Disable gradient computation because evaluation only needs forward passes.
    # This reduces overhead and makes the intent of the loop explicit.
    with torch.no_grad():
        # -------------------------------------------------------------
        # Test batch loop
        # -------------------------------------------------------------
        #
        # Each iteration processes one batch from the test set and updates the
        # running accuracy counters.
        for images, labels in test_loader:
            # Move the current test batch to the selected device.
            images = images.to(device)
            labels = labels.to(device)

            # Run the model forward on the test images.
            outputs = model(images)

            # Convert raw class scores into predicted class indices by taking
            # the highest-scoring class for each sample.
            predictions = outputs.argmax(dim=1)

            # Count how many predictions in this batch match the true labels.
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Convert the final correct-count ratio into percentage form.
    accuracy = 100.0 * correct_predictions / total_samples
    return accuracy