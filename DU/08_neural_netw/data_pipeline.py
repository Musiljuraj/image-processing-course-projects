# This module is the data layer of the whole GTSRB assignment.
# It sits below the top-level experiment controller in main.py and is
# responsible for preparing a consistent input pipeline that both compared
# models will reuse.
#
# In the runtime flow, this module performs:
#
#     raw GTSRB dataset access
#         -> image preprocessing definition
#         -> train/test dataset construction
#         -> DataLoader wrapping
#         -> shared batched input stream for both models
#
# A key design idea in this module is separation of concerns:
# - this file handles only the data side of the project,
# - it does not define models,
# - it does not train models,
# - it does not evaluate models.
#
# That separation is important because both the MLP and CNN must receive the
# same underlying data in the same basic format. Centralizing the data logic
# here makes that consistency easy to maintain.

"""
Dataset loading and preprocessing for the GTSRB assignment.

Current implementation stage:
- REAL data loading is implemented
- REAL preprocessing is implemented
- REAL DataLoaders are created

This module is responsible only for the data part of the assignment.

It should NOT contain:
- model definitions
- training loop
- evaluation logic

Why this module exists:
The project compares two model architectures on the same traffic-sign task.
To make that comparison fair, both models need to reuse exactly the same data
pipeline:
- the same dataset source,
- the same train/test split provided by torchvision,
- the same image resizing,
- the same tensor conversion,
- the same batching interface.

This file therefore acts as the shared data-entry layer for the whole
experiment.
"""

# DataLoader is the standard PyTorch wrapper that turns datasets into iterable
# mini-batches for training and evaluation.
from torch.utils.data import DataLoader

# torchvision.transforms provides the preprocessing pipeline applied to every
# loaded image before it is handed to the model.
from torchvision import transforms

# GTSRB is the built-in torchvision dataset class for the German Traffic Sign
# Recognition Benchmark used in this assignment.
from torchvision.datasets import GTSRB


# ---------------------------------------------------------------------
# Central data-stage configuration
# ---------------------------------------------------------------------
#
# These constants define the shared defaults for the data pipeline.
# Grouping them here keeps the rest of the module cleaner and makes it easy to
# understand which basic assumptions the whole input pipeline is built around.
# ---------------------------------------------------------------------

# Root directory where the GTSRB dataset will be stored or downloaded.
DATA_ROOT = "./data"

# Fixed target image size used by both models.
# Resizing is needed because the original dataset images do not all share the
# same spatial resolution.
IMAGE_SIZE = (32, 32)

# Default batch size used when the caller does not explicitly request another
# value.
DEFAULT_BATCH_SIZE = 32


def get_transforms() -> transforms.Compose:
    """
    Create and return the image preprocessing pipeline.

    The current preprocessing is intentionally minimal and contains only what
    the rest of the project absolutely needs:
    1. resize every image to a fixed 32x32 resolution,
    2. convert every image into a PyTorch tensor.

    Why this preprocessing is enough for the current project:
    - the models need a consistent input size,
    - PyTorch models and DataLoaders operate on tensors, not PIL images,
    - keeping preprocessing simple makes the baseline comparison easier to
      interpret.

    The function returns one reusable torchvision Compose object so the same
    preprocessing can be attached to both the training and test datasets.
    """

    # Build the preprocessing pipeline in the exact order in which each image
    # should be transformed before entering the model.
    return transforms.Compose(
        [
            # Step 1:
            # force every image into the same spatial resolution so the models
            # always receive a fixed-size input.
            transforms.Resize(IMAGE_SIZE),

            # Step 2:
            # convert the resized image into a PyTorch tensor so it can be
            # batched by DataLoader and processed by the neural networks.
            transforms.ToTensor(),
        ]
    )


def get_datasets(data_root: str = DATA_ROOT) -> tuple[GTSRB, GTSRB]:
    """
    Create and return the training and test GTSRB datasets.

    Parameters:
    - data_root:
        root directory where the dataset is expected to exist or where it
        should be downloaded

    Behavior:
    - one shared preprocessing pipeline is created first,
    - the training split is built with that preprocessing attached,
    - the test split is built with the same preprocessing attached,
    - download=True allows automatic dataset download when files are missing.

    Important design detail:
    This function returns dataset objects only.
    It does not wrap them into DataLoaders yet, because dataset construction
    and batching are treated as two separate steps in this module.
    """

    # Create the shared preprocessing pipeline once so both train and test
    # splits use exactly the same input transformation logic.
    transform = get_transforms()

    # Build the training dataset.
    # torchvision handles the actual dataset download/reuse internally.
    train_dataset = GTSRB(
        root=data_root,
        split="train",
        transform=transform,
        download=True,
    )

    # Build the test dataset using the same transform so evaluation is performed
    # on inputs that follow the same basic preprocessing rules.
    test_dataset = GTSRB(
        root=data_root,
        split="test",
        transform=transform,
        download=True,
    )

    # Return the raw dataset objects so the caller or the next helper function
    # can decide how they should be batched.
    return train_dataset, test_dataset


def get_data_loaders(
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_root: str = DATA_ROOT,
) -> tuple[DataLoader, DataLoader]:
    """
    Create and return training and test DataLoaders.

    Parameters:
    - batch_size:
        number of samples per batch
    - data_root:
        directory where the GTSRB dataset is stored or downloaded

    High-level flow:
    1. build the train and test dataset objects,
    2. wrap the training dataset in a shuffled DataLoader,
    3. wrap the test dataset in a non-shuffled DataLoader,
    4. return both loaders.

    Design choices used here:
    - shuffle=True for the training loader:
        training batches should be randomized to reduce order effects
    - shuffle=False for the test loader:
        evaluation should use a stable deterministic dataset order
    - num_workers=0:
        this is the safest simple baseline for student code and avoids
        cross-platform multiprocessing issues

    Return:
    - train_loader
    - test_loader

    Why this helper exists:
    main.py should not need to know the details of dataset creation, transform
    attachment, or DataLoader setup. It should be able to ask for the shared
    train/test input streams in one call.
    """

    # First create the raw dataset objects. This keeps dataset construction
    # centralized in one helper and batching logic centralized in this helper.
    train_dataset, test_dataset = get_datasets(data_root=data_root)

    # Wrap the training dataset into a DataLoader that shuffles samples each
    # epoch, which is the standard choice for stochastic training.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Wrap the test dataset into a DataLoader that preserves sample order.
    # Test data is not used for parameter updates, so shuffling is unnecessary.
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Return both loaders as the final shared data interface used by main.py
    # and, indirectly, by both compared models.
    return train_loader, test_loader