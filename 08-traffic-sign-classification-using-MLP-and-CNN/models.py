# This module is the architecture layer of the whole GTSRB assignment.
# It sits between the shared data pipeline and the shared training/evaluation
# logic and is responsible only for defining the neural network structures that
# will later be trained and compared.
#
# In the project flow, this module provides:
#
#     preprocessed 3 x 32 x 32 image batch
#         -> MLPClassifier forward pass
#             or
#         -> CNNClassifier forward pass
#         -> raw class scores (logits) for 43 traffic-sign classes
#
# A key idea in this module is architectural separation:
# - data_pipeline.py handles how images are loaded and prepared,
# - this file handles how those images are transformed into predictions,
# - train_eval.py handles how model parameters are optimized and evaluated.
#
# That separation makes the experiment easier to understand because each module
# has one clear role. This file is specifically the place where the two model
# baselines are defined side by side under a shared task setup.

"""
Model definitions for the GTSRB assignment.

Current implementation stage:
- REAL MLP model is implemented
- REAL CNN model is implemented
- both models can already process one input batch

This module is responsible only for model architecture.

It should NOT contain:
- dataset loading
- training loop
- evaluation loop

Why this module exists:
The assignment compares two different neural-network approaches on the same
traffic-sign classification task:
- a dense fully connected baseline,
- a convolutional neural network baseline.

To keep that comparison clean, the model architectures are isolated here from
the rest of the project. That way:
- the architectures are easy to inspect,
- both models can be trained by the same shared training code,
- main.py can instantiate and compare them without mixing model structure with
  data or optimization logic.
"""

# torch is needed here mainly for tensor type annotations in forward methods.
import torch

# nn provides the neural-network building blocks used to define both model
# architectures.
import torch.nn as nn


# ---------------------------------------------------------------------
# Shared task-level constants
# ---------------------------------------------------------------------
#
# These constants define the common assumptions both models rely on:
# - number of target classes,
# - number of image channels,
# - fixed image size after preprocessing,
# - flattened input size needed by the MLP baseline.
#
# Centralizing them here keeps both architecture definitions consistent and
# makes the task setup explicit at the top of the module.
# ---------------------------------------------------------------------

# The GTSRB task contains 43 traffic-sign categories.
NUM_CLASSES = 43

# Input images are RGB, so each image has 3 channels.
INPUT_CHANNELS = 3

# The shared preprocessing pipeline in data_pipeline.py resizes every image to
# 32 x 32 pixels.
IMAGE_SIZE = 32

# The MLP receives the full image as one flattened vector, so its input size is
# channels * height * width.
FLATTENED_INPUT_SIZE = INPUT_CHANNELS * IMAGE_SIZE * IMAGE_SIZE


class MLPClassifier(nn.Module):
    """
    Fully connected baseline model for traffic-sign classification.

    High-level design:
    - input image shape: 3 x 32 x 32
    - the image is flattened into one vector of length 3072
    - one hidden linear layer learns a dense intermediate representation
    - ReLU introduces non-linearity
    - one final linear layer produces 43 raw class scores

    Why this model exists:
    This is the simple dense baseline in the experiment. It gives a reference
    point for comparison against the CNN. The important limitation of this
    architecture is that once the image is flattened, local spatial structure
    is no longer represented explicitly.

    Important note:
    - there is NO softmax layer at the end
    - that is intentional because CrossEntropyLoss later expects raw logits
    """

    def __init__(self, hidden_size: int = 256) -> None:
        """
        Initialize the MLP architecture.

        Parameters:
        - hidden_size:
            number of neurons in the hidden fully connected layer

        The network is stored as one Sequential container because the model is
        structurally simple and follows a strictly linear layer-by-layer flow.
        """
        super().__init__()

        # Build the dense baseline as one straightforward feed-forward stack:
        #
        # image batch
        #     -> flatten each image into one 1D vector
        #     -> project into hidden feature space
        #     -> apply ReLU non-linearity
        #     -> project to 43 output logits
        self.network = nn.Sequential(
            # Convert [batch_size, 3, 32, 32] into [batch_size, 3072].
            nn.Flatten(),

            # First dense transformation from raw flattened pixels into a
            # learned hidden representation.
            nn.Linear(FLATTENED_INPUT_SIZE, hidden_size),

            # ReLU gives the model non-linear expressive power.
            nn.ReLU(),

            # Final dense layer maps the hidden representation to one raw score
            # per traffic-sign class.
            nn.Linear(hidden_size, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass through the MLP.

        Input:
        - x:
            batch of images with shape [batch_size, 3, 32, 32]

        Output:
        - tensor of shape [batch_size, 43]
          containing raw class scores for each image

        The forward path itself stays very small because the architectural logic
        is already fully described by self.network.
        """

        # Pass the whole batch through the Sequential architecture and return
        # the resulting logits.
        return self.network(x)


class CNNClassifier(nn.Module):
    """
    Convolutional neural network baseline for traffic-sign classification.

    High-level design:
    - two convolutional feature-extraction blocks,
    - each block applies convolution + ReLU + max pooling,
    - the extracted spatial feature maps are then flattened,
    - one hidden fully connected layer is used before the final classifier
      output layer.

    Shape progression:
    - input:           3 x 32 x 32
    - after conv1:    16 x 32 x 32
    - after pool1:    16 x 16 x 16
    - after conv2:    32 x 16 x 16
    - after pool2:    32 x  8 x  8
    - flattened:      32 * 8 * 8 = 2048 features

    Why this model exists:
    This is the convolutional baseline in the experiment. Unlike the MLP, it
    preserves local spatial structure through convolution and pooling, which is
    usually more suitable for image classification tasks.

    Important note:
    - there is NO softmax layer at the end
    - that is intentional because CrossEntropyLoss later expects raw logits
    """

    def __init__(self, hidden_size: int = 128) -> None:
        """
        Initialize the CNN architecture.

        Parameters:
        - hidden_size:
            number of neurons in the hidden fully connected layer after the
            convolutional feature extractor

        The architecture is split into two conceptual parts:
        - self.features:
            convolutional feature extraction
        - self.classifier:
            dense classification head operating on flattened features
        """
        super().__init__()

        # -----------------------------------------------------------------
        # Convolutional feature extractor
        # -----------------------------------------------------------------
        #
        # This part keeps the image in spatial form while learning local
        # features. Pooling gradually reduces spatial resolution and keeps the
        # most relevant activations.
        self.features = nn.Sequential(
            # First convolution block:
            # learn 16 local feature maps directly from the 3-channel input.
            nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=3, padding=1),

            # Add non-linearity after the convolution.
            nn.ReLU(),

            # Downsample spatial resolution from 32x32 to 16x16.
            nn.MaxPool2d(kernel_size=2),

            # Second convolution block:
            # learn a richer set of 32 feature maps from the previous 16 maps.
            nn.Conv2d(16, 32, kernel_size=3, padding=1),

            # Again apply non-linearity after convolution.
            nn.ReLU(),

            # Downsample spatial resolution from 16x16 to 8x8.
            nn.MaxPool2d(kernel_size=2),
        )

        # -----------------------------------------------------------------
        # Dense classifier head
        # -----------------------------------------------------------------
        #
        # After the convolutional feature extractor has produced compact spatial
        # feature maps, this head converts them into final class logits.
        self.classifier = nn.Sequential(
            # Flatten [batch_size, 32, 8, 8] into [batch_size, 2048].
            nn.Flatten(),

            # Project the flattened convolutional representation into a hidden
            # dense representation.
            nn.Linear(32 * 8 * 8, hidden_size),

            # Add non-linearity before the final class projection.
            nn.ReLU(),

            # Final dense layer producing one raw score for each class.
            nn.Linear(hidden_size, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass through the CNN.

        Input:
        - x:
            batch of images with shape [batch_size, 3, 32, 32]

        Output:
        - tensor of shape [batch_size, 43]
          containing raw class scores for each image

        The forward flow is intentionally split into two explicit stages:
        - spatial feature extraction,
        - dense classification from extracted features.
        """

        # Stage 1:
        # transform the input batch into learned convolutional feature maps that
        # preserve useful local image structure.
        x = self.features(x)

        # Stage 2:
        # flatten and classify those extracted features into final logits.
        x = self.classifier(x)

        return x