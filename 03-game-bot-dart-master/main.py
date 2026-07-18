"""
Entry point of the whole bot.

This module does not do any image processing, matching, prediction, or clicking by itself.
Its job is to prepare startup conditions for the rest of the system and then hand control
to the main runtime controller.

The startup flow is intentionally simple and linear:

1. Build a command-line argument parser so runtime behavior can be adjusted without editing code.
2. Read the actual CLI arguments provided by the user.
3. Create one default top-level configuration object (BotConfig).
4. Apply CLI overrides on top of those default values.
5. Optionally print a startup summary so the active runtime configuration is visible.
6. Create the main controller object.
7. Start the controller loop.

This makes main.py the "assembly layer" of the project:
it wires together configuration, command-line input, and the controller,
but leaves all actual operational work to lower-level modules.
"""

import argparse  # Standard library module for parsing command-line arguments like --threshold or --quiet
import sys       # Standard library module; used here for printing errors to stderr

from bot.config import BotConfig                  # Top-level configuration object for the whole bot
from bot.controller import DartMasterController   # Main loop controller that runs the bot


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the command-line argument parser.

    This function defines every runtime option that can be changed from the terminal
    without touching the source code. The parser is built only once during startup,
    then used by main() to read the command line into a structured argparse.Namespace.

    Conceptually, these options fall into a few groups:

    - bridge communication options
    - matching sensitivity options
    - click timing options
    - optional motion-prediction options
    - debug output options
    - startup verbosity options

    Keeping parser construction in its own function keeps main() shorter and makes the
    CLI definition easy to inspect in one place.
    """
    # Create the parser object shown when the user runs: python3 main.py --help
    # The description explains the high-level purpose of the program from the terminal user's perspective.
    parser = argparse.ArgumentParser(
        description=(
            "Dart Master bot entry point. "
            "Reads frames from the Windows bridge, performs full-frame template matching, "
            "and can optionally use short-history predictive clicking."
        )
    )

    # Optional override for the HTTP base URL of the Windows bridge.
    # This is useful when the bridge is running on a different host/port than the default.
    parser.add_argument(
        "--bridge-base-url",
        type=str,          # value should be parsed as text
        default=None,      # None means: do not override the config default
        help="Override Windows bridge base URL, e.g. http://127.0.0.1:8080",
    )

    # Optional override for template matching threshold.
    # This changes how strong a match must be before the controller allows a click.
    parser.add_argument(
        "--threshold",
        type=float,        # value should be parsed as a floating-point number
        default=None,      # None means: keep the default threshold from config
        help="Override template match threshold, e.g. 0.72",
    )

    # Optional override for click cooldown time.
    # This controls how long the bot waits after one click before another click can happen.
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Override click cooldown in seconds, e.g. 0.50",
    )

    # Boolean flag: if present, enable predictive clicking.
    # Without this flag, the bot keeps baseline behavior and clicks the detected center directly.
    parser.add_argument(
        "--enable-prediction",
        action="store_true",  # absent -> False, present -> True
        help="Enable short-history predictive clicking.",
    )

    # Optional override for how many recent center points are kept in motion history.
    # This controls the length of the short movement memory used by prediction.
    parser.add_argument(
        "--prediction-history-size",
        type=int,
        default=None,
        help="Override motion history size, e.g. 4",
    )

    # Optional override for minimum number of history points required before prediction can be used.
    # This prevents prediction from activating on too little movement evidence.
    parser.add_argument(
        "--prediction-min-history",
        type=int,
        default=None,
        help="Minimum number of history points required before prediction is allowed.",
    )

    # Optional override for the score required before a match is trusted enough to update motion history.
    # This separates "good enough to learn motion from" from "good enough to click."
    parser.add_argument(
        "--prediction-update-threshold",
        type=float,
        default=None,
        help="Minimum score required for a match to update motion history.",
    )

    # Optional override for minimum motion magnitude required to treat movement as real motion, not jitter.
    # This helps prevent the prediction logic from reacting to tiny noisy fluctuations.
    parser.add_argument(
        "--prediction-min-motion",
        type=float,
        default=None,
        help="Minimum recent motion (px) to treat as real motion instead of jitter.",
    )

    # Optional override for how consistent recent movement direction must be.
    # This controls how strict the prediction logic is about requiring stable motion direction.
    parser.add_argument(
        "--prediction-direction-consistency",
        type=float,
        default=None,
        help="Minimum average direction consistency for prediction, e.g. 0.80",
    )

    # Optional override for how far ahead the predictive click should aim.
    # Larger values push the click farther along the recent motion direction.
    parser.add_argument(
        "--prediction-scale",
        type=float,
        default=None,
        help="How far ahead to click relative to recent average motion, e.g. 0.35",
    )

    # Optional override for the maximum safety-capped prediction jump in pixels.
    # Even if estimated movement is large, prediction will not jump beyond this cap.
    parser.add_argument(
        "--prediction-max-jump",
        type=int,
        default=None,
        help="Maximum forward prediction jump in pixels.",
    )

    # Optional override for how many weak/missing frames are tolerated before motion history is forgotten.
    # This prevents stale old movement from being reused after tracking becomes unreliable.
    parser.add_argument(
        "--prediction-max-missed-frames",
        type=int,
        default=None,
        help="Forget motion history after this many weak/missing frames.",
    )

    # Boolean flag: if present, allow debug window display.
    # The default remains off so no visual overlay interferes with the gameplay area.
    parser.add_argument(
        "--show-window",
        action="store_true",
        help=(
            "Enable debug window display. "
            "By default this is OFF so no image window overlays the game."
        ),
    )

    # Boolean flag: if present, save annotated debug frames to disk.
    # This is mainly useful for post-run inspection and tuning.
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save annotated debug frames to disk.",
    )

    # Boolean flag: if present, suppress startup summary from main.py.
    # The controller may still print runtime loop status later.
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce startup prints from main.py (controller still prints loop status).",
    )

    # Return the fully configured parser object.
    # No parsing happens here yet; this function only defines the CLI structure.
    return parser


def apply_cli_overrides(cfg: BotConfig, args: argparse.Namespace) -> BotConfig:
    """
    Apply command-line overrides to the default BotConfig and return the updated config.

    The important idea here is that the project has one canonical configuration object,
    but the terminal user is allowed to override selected values at startup.

    The logic is intentionally explicit and field-by-field:
    each possible override is checked separately, validated, and then copied into cfg.

    This function does not create a new independent config structure from scratch.
    Instead, it starts from the already constructed default BotConfig and mutates only
    the fields that were explicitly overridden on the command line.

    That approach has two practical benefits:
    - defaults stay centralized in config.py
    - CLI code only needs to describe differences from the defaults
    """
    # If the user provided a bridge URL, replace the default config value.
    # This affects all later HTTP communication because lower-level bridge clients use cfg.bridge.base_url.
    if args.bridge_base_url is not None:
        cfg.bridge.base_url = args.bridge_base_url

    # If the user provided a match threshold, validate it and store it.
    # Thresholds are normalized scores, so the allowed interval is 0.0 to 1.0.
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            raise ValueError("--threshold must be between 0.0 and 1.0")
        cfg.match.threshold = args.threshold

    # If the user provided a cooldown, validate it and store it.
    # Negative cooldown would make no sense, so it is rejected immediately.
    if args.cooldown is not None:
        if args.cooldown < 0:
            raise ValueError("--cooldown must be non-negative")
        cfg.click.cooldown_seconds = args.cooldown

    # Enable prediction only if the CLI flag is present.
    # This preserves the baseline non-predictive behavior unless the user explicitly opts in.
    if args.enable_prediction:
        cfg.prediction.enabled = True

    # Override prediction history size if provided.
    # At least 2 points are needed to form 1 motion step, so smaller values are invalid.
    if args.prediction_history_size is not None:
        if args.prediction_history_size < 2:
            raise ValueError("--prediction-history-size must be at least 2")
        cfg.prediction.history_size = args.prediction_history_size

    # Override minimum history required before prediction is allowed.
    # Again, fewer than 2 points cannot define motion at all.
    if args.prediction_min_history is not None:
        if args.prediction_min_history < 2:
            raise ValueError("--prediction-min-history must be at least 2")
        cfg.prediction.min_history = args.prediction_min_history

    # Override prediction update threshold if provided.
    # This is also a normalized confidence-like value, so it must stay in [0.0, 1.0].
    if args.prediction_update_threshold is not None:
        if not (0.0 <= args.prediction_update_threshold <= 1.0):
            raise ValueError("--prediction-update-threshold must be between 0.0 and 1.0")
        cfg.prediction.update_threshold = args.prediction_update_threshold

    # Override minimum motion threshold if provided.
    # Negative motion threshold has no meaning, so it is rejected.
    if args.prediction_min_motion is not None:
        if args.prediction_min_motion < 0:
            raise ValueError("--prediction-min-motion must be non-negative")
        cfg.prediction.min_motion_px = args.prediction_min_motion

    # Override minimum direction consistency if provided.
    # Valid range is -1.0 to 1.0 because direction agreement scores naturally live in that interval:
    # -1 means opposite direction, 0 means unrelated/perpendicular, +1 means same direction.
    if args.prediction_direction_consistency is not None:
        if not (-1.0 <= args.prediction_direction_consistency <= 1.0):
            raise ValueError("--prediction-direction-consistency must be between -1.0 and 1.0")
        cfg.prediction.min_direction_consistency = args.prediction_direction_consistency

    # Override how far ahead prediction aims.
    # Negative forward scaling would invert the meaning of prediction, so it is not allowed.
    if args.prediction_scale is not None:
        if args.prediction_scale < 0:
            raise ValueError("--prediction-scale must be non-negative")
        cfg.prediction.prediction_scale = args.prediction_scale

    # Override maximum allowed forward prediction jump.
    # This is a safety cap, so it must be zero or positive.
    if args.prediction_max_jump is not None:
        if args.prediction_max_jump < 0:
            raise ValueError("--prediction-max-jump must be non-negative")
        cfg.prediction.max_prediction_px = args.prediction_max_jump

    # Override how many weak/missing frames are allowed before forgetting motion history.
    # Negative counts are invalid because this setting represents a number of tolerated failures.
    if args.prediction_max_missed_frames is not None:
        if args.prediction_max_missed_frames < 0:
            raise ValueError("--prediction-max-missed-frames must be non-negative")
        cfg.prediction.max_missed_frames = args.prediction_max_missed_frames

    # Copy the debug-window CLI flag into config.
    # bool(...) is slightly redundant here because store_true already gives a boolean,
    # but it makes the intention explicit: the config field should definitely contain True/False.
    cfg.debug.show_window = bool(args.show_window)

    # Enable saving debug frames only if the CLI flag is present.
    # This is asymmetric on purpose: no flag means "leave default as-is", flag means force-enable.
    if args.save_debug_frames:
        cfg.debug.save_debug_frames = True

    # Return the updated configuration object.
    # Returning cfg keeps the function easy to compose in main(), even though cfg was mutated in place.
    return cfg


def print_startup_summary(cfg: BotConfig) -> None:
    """
    Print the final active startup configuration in a human-readable form.

    This summary is intentionally operational rather than technical:
    it shows the effective values the bot will actually use after defaults and CLI overrides
    have already been merged together.

    That makes it easy to verify, before the main loop starts, that the runtime settings
    are really what was intended.
    """
    # Print a short banner so the beginning of bot execution is obvious in terminal logs.
    print("Starting Dart Master bot")

    # Print the bridge location the bot will communicate with.
    print(f"Bridge base URL: {cfg.bridge.base_url}")

    # Print all template paths after they have been resolved into the config.
    # Converting each Path to str keeps the output terminal-friendly.
    print(f"Templates: {[str(path) for path in cfg.templates.paths]}")

    # Print whether matching is performed in grayscale mode.
    print(f"Use grayscale: {cfg.templates.use_gray}")

    # Print the final detection threshold that controls click eligibility.
    print(f"Match threshold: {cfg.match.threshold}")

    # Print click cooldown so runtime click pacing is visible.
    print(f"Click cooldown: {cfg.click.cooldown_seconds}")

    # Print the full prediction-related configuration group.
    # This is especially useful because prediction has multiple interacting safety settings.
    print(f"Prediction enabled: {cfg.prediction.enabled}")
    print(f"Prediction history size: {cfg.prediction.history_size}")
    print(f"Prediction min history: {cfg.prediction.min_history}")
    print(f"Prediction update threshold: {cfg.prediction.update_threshold}")
    print(f"Prediction min motion px: {cfg.prediction.min_motion_px}")
    print(f"Prediction direction consistency: {cfg.prediction.min_direction_consistency}")
    print(f"Prediction scale: {cfg.prediction.prediction_scale}")
    print(f"Prediction max jump px: {cfg.prediction.max_prediction_px}")
    print(f"Prediction max missed frames: {cfg.prediction.max_missed_frames}")

    # Print debug-related settings so it is clear whether any visual/debug artifacts will be produced.
    print(f"Show debug window: {cfg.debug.show_window}")
    print(f"Save debug frames: {cfg.debug.save_debug_frames}")

    # Final reminder about how the runtime loop is normally stopped.
    print("Stop the bot with Ctrl+C.")


def main() -> int:
    """
    Main startup routine:
    - build parser
    - read CLI args
    - build default config
    - apply CLI overrides
    - optionally print startup summary
    - create controller
    - run controller
    - return process exit code

    main() is intentionally small and orchestration-focused.
    It does not contain business logic for matching, prediction, or clicking.
    Its main responsibility is to convert startup conditions into a running controller.

    Return value convention:
    - 0 means normal success / normal manual stop
    - 1 means startup or runtime error escaped to this level
    """
    # Build the command-line parser and read the actual command line.
    # After parse_args(), all CLI inputs are available as attributes on args.
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        # Start from default config values defined in config.py.
        # This creates one structured config tree that the entire application will share.
        cfg = BotConfig()

        # Apply any command-line overrides on top of the defaults.
        # This preserves config.py as the source of defaults while still allowing quick runtime tuning.
        cfg = apply_cli_overrides(cfg, args)

        # Unless quiet mode is enabled, print the final active settings.
        # This happens after overrides so the summary reflects the real effective runtime config.
        if not args.quiet:
            print_startup_summary(cfg)

        # Create the main controller object and start the bot loop.
        # From this point onward, control effectively moves into controller.py.
        controller = DartMasterController(cfg)
        controller.run()

        # Return success exit code.
        # Reaching this line means the controller finished without an unhandled error.
        return 0

    except KeyboardInterrupt:
        # Ctrl+C from the user is treated as a normal manual stop, not a crash.
        # This keeps the terminal exit status clean during intentional shutdown.
        print("\nStopped by user.")
        return 0

    except Exception as exc:
        # Any other exception is treated as an error.
        # The message is printed to stderr so it is clearly separated from normal stdout logging.
        # Returning non-zero exit code signals failure to the operating system / shell.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1



if __name__ == "__main__":
    # Run main() only when this file is executed as the program entry script.
    # main() returns an integer exit code; SystemExit passes it to the operating system.
    raise SystemExit(main())