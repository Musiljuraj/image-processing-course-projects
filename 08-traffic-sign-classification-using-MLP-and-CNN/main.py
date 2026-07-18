# This module is the top orchestration layer of the whole GTSRB assignment.
# It sits above the lower project modules and is responsible for connecting
# them into one complete, end-to-end experiment.
#
# In the runtime flow, this module performs:
#
#     experiment start
#         -> device selection
#         -> shared data loading
#         -> shared loss-function setup
#         -> MLP training / timing / evaluation
#         -> CNN training / timing / evaluation
#         -> final side-by-side comparison
#         -> brief interpretation of the outcome
#
# A key idea in this module is fairness of comparison:
# - both models use the same data source,
# - both models use the same preprocessing,
# - both models use the same batch size,
# - both models use the same epoch count,
# - both models use the same optimizer type,
# - both models use the same learning rate,
# - both models are evaluated on the same test split.
#
# That means this file is not just "the place that runs everything".
# It is specifically the place that controls a fair comparison between two
# baseline architectures under matched experimental conditions.

"""
Main entry point for the GTSRB assignment.

Current implementation stage:
- data_pipeline.py is implemented and reused from here
- models.py is implemented and instantiated from here
- train_eval.py is implemented and reused from here
- training-time measurement is performed here
- final comparison summary is produced here

Purpose of this module:
- run the full experiment end-to-end,
- train both models under the same conditions,
- evaluate both models on the test set,
- measure training time for both models,
- print one final comparison of loss, accuracy, and speed.

Why this module exists:
The lower modules already separate the main building blocks:
- data_pipeline.py handles dataset loading and preprocessing,
- models.py defines the MLP and CNN architectures,
- train_eval.py provides shared training and evaluation logic.

The job of this file is to connect those already-separated parts into one
complete experimental run. In other words, this is the control-flow layer of
the project.

Important note:
- this version is already suitable as a minimal final assignment solution,
- the comparison remains fair because both models use exactly the same overall
  experimental setup except for the model architecture itself.
"""

# time is used only for precise training-time measurement of both models.
# That timing information later becomes part of the final comparison summary.
import time

# torch is needed here mainly for:
# - device selection,
# - CUDA availability checks,
# - optional CUDA synchronization before/after timing.
import torch

# nn is used here to instantiate the shared loss function.
# The actual model architectures are defined elsewhere in models.py.
import torch.nn as nn

# optim is used here to create one optimizer per model.
# The training loop itself is not implemented here; it is delegated to
# train_eval.py.
import torch.optim as optim

# The data layer provides shared train/test DataLoaders for both models.
from data_pipeline import get_data_loaders

# The architecture layer provides the two models being compared.
from models import CNNClassifier, MLPClassifier

# The shared training/evaluation layer provides one reusable procedure for both
# models, which helps keep the comparison methodologically consistent.
from train_eval import evaluate_model, train_model

# ---------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------
#
# These constants define the common experiment settings reused by both models.
# Grouping them here makes the comparison setup explicit and easy to change in
# one place if needed.
#
# The important idea is that these values should remain identical for both
# architectures. That way, the experiment isolates the effect of model design
# rather than mixing in changes to optimizer settings or data batching.
# ---------------------------------------------------------------------

# Number of samples processed in one batch.
BATCH_SIZE = 32

# Number of full passes through the training set.
EPOCHS = 10

# Shared Adam learning rate for both model runs.
LEARNING_RATE = 0.001


def synchronize_if_needed(device: torch.device) -> None:
    """
    Synchronize CUDA before or after timing if the experiment runs on GPU.

    Why this helper exists:
    CUDA operations are often asynchronous, which means Python timing calls can
    otherwise measure only when work was scheduled, not when it was actually
    finished.

    On CPU, no synchronization is needed, so the function becomes a no-op.
    """

    # Timing accuracy matters in this project because training speed is part of
    # the final comparison. On CUDA, we therefore force all queued GPU work to
    # finish before the timer is read.
    if device.type == "cuda":
        torch.cuda.synchronize()


def format_seconds(seconds: float) -> str:
    """
    Format elapsed time into a short human-readable string.

    This helper exists only to keep the print statements in main() cleaner and
    to ensure both model timings are displayed in the same format.
    """

    # Standardize time formatting in one place so later reporting code stays
    # simple and consistent.
    return f"{seconds:.2f} s"


def main() -> None:
    """
    Run the full GTSRB experiment.

    High-level flow:
    1. select the compute device,
    2. load the shared train/test data,
    3. define the shared loss function,
    4. train, time, and evaluate the MLP baseline,
    5. train, time, and evaluate the CNN baseline,
    6. compare both finished runs and print the final interpretation.

    The most important structural idea in this function is that it does not
    implement the lower-level logic itself. It reuses:
    - data_pipeline.py for data,
    - models.py for architectures,
    - train_eval.py for training/evaluation.

    That keeps this file focused on experiment control and comparison logic.
    """

    # Print a clear header so the console output immediately shows that this is
    # one complete final experiment run rather than an isolated smoke test.
    print("=" * 60)
    print("GTSRB assignment - final training and comparison run")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Step 1: device selection
    # -----------------------------------------------------------------
    #
    # The experiment can run either on:
    # - CUDA GPU, if available,
    # - or CPU otherwise.
    #
    # This decision is centralized here because all later training and
    # evaluation calls should reuse the same selected device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()

    # -----------------------------------------------------------------
    # Step 2: shared data loading
    # -----------------------------------------------------------------
    #
    # Both architectures must be trained and evaluated on the exact same data
    # splits. That is why the loaders are created once here and then reused for
    # both the MLP and CNN experiments.
    train_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

    # Print a compact overview of the shared experimental setup so the console
    # output documents what conditions were actually used for the run.
    print("Data loaders created successfully.")
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size:  {len(test_loader.dataset)}")
    print(f"Batch size:         {BATCH_SIZE}")
    print(f"Epochs:             {EPOCHS}")
    print(f"Learning rate:      {LEARNING_RATE}")
    print()

    # -----------------------------------------------------------------
    # Step 3: shared loss-function setup
    # -----------------------------------------------------------------
    #
    # CrossEntropyLoss is the standard loss for multi-class classification.
    # In this project it matches the models because they output raw logits for
    # 43 traffic-sign classes.
    criterion = nn.CrossEntropyLoss()

    # ================================================================
    # MLP experiment
    # ================================================================
    #
    # This is the first model run in the comparison.
    # The MLP serves as the dense baseline:
    # - it receives the same images,
    # - uses the same optimizer type,
    # - uses the same learning rate,
    # - trains for the same number of epochs,
    # - and is evaluated by the same evaluation function.
    print("=" * 60)
    print("Training MLPClassifier")
    print("=" * 60)

    # Create the MLP architecture and its optimizer.
    # The optimizer is attached only to this model's parameters, so the MLP and
    # CNN are trained completely independently.
    mlp_model = MLPClassifier()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)

    # Synchronize before timing on CUDA so the timer starts only after any
    # earlier queued GPU work has fully finished.
    synchronize_if_needed(device)
    mlp_start_time = time.perf_counter()

    # Reuse the shared training loop. This keeps the comparison fair because
    # both architectures go through the same training procedure.
    mlp_losses = train_model(
        model=mlp_model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=mlp_optimizer,
        device=device,
        epochs=EPOCHS,
    )

    # Synchronize again before reading the stop time on CUDA so the recorded
    # duration reflects the real completed training time.
    synchronize_if_needed(device)
    mlp_training_time = time.perf_counter() - mlp_start_time

    # After training is complete, evaluate the MLP on the shared test set.
    mlp_accuracy = evaluate_model(
        model=mlp_model,
        test_loader=test_loader,
        device=device,
    )

    # Report the final MLP results.
    # The last item in mlp_losses is the average training loss from the final
    # epoch, which is used here as the final training-loss summary value.
    print(f"MLP final training loss: {mlp_losses[-1]:.4f}")
    print(f"MLP test accuracy:       {mlp_accuracy:.2f}%")
    print(f"MLP training time:       {format_seconds(mlp_training_time)}")
    print(f"MLP average time/epoch:  {format_seconds(mlp_training_time / EPOCHS)}")
    print()

    # ================================================================
    # CNN experiment
    # ================================================================
    #
    # This is the second model run in the comparison.
    # It reuses the same global experiment setup as the MLP, but swaps in a CNN
    # architecture. That makes the architecture itself the main variable being
    # compared.
    print("=" * 60)
    print("Training CNNClassifier")
    print("=" * 60)

    # Create the CNN architecture and its own optimizer.
    # Like the MLP run above, this is a fresh model with its own parameter set.
    cnn_model = CNNClassifier()
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)

    # Start timing the CNN training under the same timing rules used for the
    # MLP run.
    synchronize_if_needed(device)
    cnn_start_time = time.perf_counter()

    # Reuse the exact same shared training procedure.
    cnn_losses = train_model(
        model=cnn_model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=cnn_optimizer,
        device=device,
        epochs=EPOCHS,
    )

    # Stop timing only after all CUDA work is complete, for consistency with
    # the MLP measurement.
    synchronize_if_needed(device)
    cnn_training_time = time.perf_counter() - cnn_start_time

    # Evaluate the trained CNN on the same shared test set.
    cnn_accuracy = evaluate_model(
        model=cnn_model,
        test_loader=test_loader,
        device=device,
    )

    # Report the final CNN results in the same format as the MLP results so the
    # console output is easy to compare visually.
    print(f"CNN final training loss: {cnn_losses[-1]:.4f}")
    print(f"CNN test accuracy:       {cnn_accuracy:.2f}%")
    print(f"CNN training time:       {format_seconds(cnn_training_time)}")
    print(f"CNN average time/epoch:  {format_seconds(cnn_training_time / EPOCHS)}")
    print()

    # ================================================================
    # Final comparison summary
    # ================================================================
    #
    # At this point both runs are finished, so the remaining job is to compute
    # a compact comparison between them.
    #
    # The three main comparison axes are:
    # - final accuracy,
    # - final training loss,
    # - training speed.
    accuracy_difference = cnn_accuracy - mlp_accuracy
    loss_difference = mlp_losses[-1] - cnn_losses[-1]
    time_difference = mlp_training_time - cnn_training_time

    # Convert raw metric comparisons into simple model labels that can be
    # reported directly in the summary section.
    better_accuracy_model = "CNNClassifier" if cnn_accuracy > mlp_accuracy else "MLPClassifier"
    lower_loss_model = "CNNClassifier" if cnn_losses[-1] < mlp_losses[-1] else "MLPClassifier"
    faster_model = "CNNClassifier" if cnn_training_time < mlp_training_time else "MLPClassifier"

    # Print the final side-by-side metric overview first.
    print("=" * 60)
    print("Final comparison")
    print("=" * 60)
    print(f"MLP final loss:         {mlp_losses[-1]:.4f}")
    print(f"CNN final loss:         {cnn_losses[-1]:.4f}")
    print(f"MLP test accuracy:      {mlp_accuracy:.2f}%")
    print(f"CNN test accuracy:      {cnn_accuracy:.2f}%")
    print(f"MLP training time:      {format_seconds(mlp_training_time)}")
    print(f"CNN training time:      {format_seconds(cnn_training_time)}")
    print()

    # Then print a more interpretive summary that explicitly states which model
    # won on each comparison axis and by how much.
    print("Comparison summary:")
    print(f"- Better final accuracy: {better_accuracy_model}")
    print(f"- Lower final loss:      {lower_loss_model}")
    print(f"- Faster training:       {faster_model}")
    print(f"- Accuracy difference:   {accuracy_difference:.2f} percentage points")
    print(f"- Loss difference:       {loss_difference:.4f}")
    print(f"- Time difference:       {time_difference:.2f} s (positive => CNN slower)")
    print()

    # -----------------------------------------------------------------
    # Brief interpretation
    # -----------------------------------------------------------------
    #
    # This final note connects the measured results back to the expected theory
    # behind the two architectures.
    #
    # The MLP acts as a dense baseline:
    # - it sees the image as one flattened vector,
    # - so explicit local spatial structure is not preserved.
    #
    # The CNN is usually more appropriate for image data:
    # - it processes local neighborhoods,
    # - preserves spatial structure more naturally,
    # - and therefore often achieves better performance on image tasks.
    #
    # The printed interpretation stays intentionally brief because the main
    # experimental evidence has already been shown above in numeric form.
    print("Interpretation:")
    print("- MLP is the simpler dense baseline.")
    print("- CNN is usually better suited for images because it preserves local spatial structure.")
    print("- The final conclusion should consider accuracy, loss, and training time together.")


# Standard Python entry-point guard.
# This ensures the experiment starts only when this file is run directly, not
# when it is imported by another module.
if __name__ == "__main__":
    main()