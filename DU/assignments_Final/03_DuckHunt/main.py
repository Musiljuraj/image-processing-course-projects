"""
Starts the bot, builds the argument parser, reads CLI options, creates one configuration object - BotConfig,
applies command-line overrides, optionally prints a startup summary,
creates DartMasterController, and starts its run() method.
""" 

import argparse  # Standard library module for parsing command-line arguments like --threshold or --quiet
import sys       # Standard library module; used here for printing errors to stderr

from bot.config import BotConfig                  # Top-level configuration object for the whole bot
from bot.controller import DartMasterController   # Main loop controller that runs the bot


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build and return the command-line argument parser.
    """
    # Create the parser object shown when the user runs: python3 main.py --help
    parser = argparse.ArgumentParser(
        description=(
            "Duck Hunt bot entry point. "  
            "Reads frames from the Windows bridge, performs multi-template multi-scale template matching, "
            "and can optionally use short-history predictive clicking."  
        )
    )

    # Optional override for the HTTP base URL of the Windows bridge
    parser.add_argument(
        "--bridge-base-url",
        type=str,          # value should be parsed as text
        default=None,      # None means: do not override the config default
        help="Override Windows bridge base URL, e.g. http://127.0.0.1:8080",
    )

    # Optional override for template matching threshold
    parser.add_argument(
        "--threshold",
        type=float,        # value should be parsed as a floating-point number
        default=None,      # None means: keep the default threshold from config
        help="Override template match threshold, e.g. 0.72",
    )

    # Optional override for click cooldown time
    parser.add_argument(
        "--cooldown",
        type=float,
        default=None,
        help="Override click cooldown in seconds, e.g. 0.50",
    )

    # Boolean flag: if present, enable predictive clicking
    parser.add_argument(
        "--enable-prediction",
        action="store_true",  # absent -> False, present -> True
        help="Enable short-history predictive clicking.",
    )

    # Optional override for how many recent center points are kept in motion history
    parser.add_argument(
        "--prediction-history-size",
        type=int,
        default=None,
        help="Override motion history size, e.g. 4",
    )

    # Optional override for minimum number of history points required before prediction can be used
    parser.add_argument(
        "--prediction-min-history",
        type=int,
        default=None,
        help="Minimum number of history points required before prediction is allowed.",
    )

    # Optional override for the score required before a match is trusted enough to update motion history
    parser.add_argument(
        "--prediction-update-threshold",
        type=float,
        default=None,
        help="Minimum score required for a match to update motion history.",
    )

    # Optional override for minimum motion magnitude required to treat movement as real motion, not jitter
    parser.add_argument(
        "--prediction-min-motion",
        type=float,
        default=None,
        help="Minimum recent motion (px) to treat as real motion instead of jitter.",
    )

    # Optional override for how consistent recent movement direction must be
    parser.add_argument(
        "--prediction-direction-consistency",
        type=float,
        default=None,
        help="Minimum average direction consistency for prediction, e.g. 0.80",
    )

    # Optional override for how far ahead the predictive click should aim
    parser.add_argument(
        "--prediction-scale",
        type=float,
        default=None,
        help="How far ahead to click relative to recent average motion, e.g. 0.35",
    )

    # Optional override for the maximum safety-capped prediction jump in pixels
    parser.add_argument(
        "--prediction-max-jump",
        type=int,
        default=None,
        help="Maximum forward prediction jump in pixels.",
    )

    # Optional override for how many weak/missing frames are tolerated before motion history is forgotten
    parser.add_argument(
        "--prediction-max-missed-frames",
        type=int,
        default=None,
        help="Forget motion history after this many weak/missing frames.",
    )

    # Boolean flag: if present, allow debug window display
    parser.add_argument(
        "--show-window",
        action="store_true",
        help=(
            "Enable debug window display. "
            "By default this is OFF so no image window overlays the game."
        ),
    )

    # Boolean flag: if present, save annotated debug frames to disk
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save annotated debug frames to disk.",
    )

    # Boolean flag: if present, suppress startup summary from main.py
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce startup prints from main.py (controller still prints loop status).",
    )

    # Return the fully configured parser object
    return parser


def apply_cli_overrides(cfg: BotConfig, args: argparse.Namespace) -> BotConfig:
    """
    Apply command-line overrides to the default BotConfig and return the updated config.
    """
    # If the user provided a bridge URL, replace the default config value
    if args.bridge_base_url is not None:
        cfg.bridge.base_url = args.bridge_base_url

    # If the user provided a match threshold, validate it and store it
    if args.threshold is not None:
        if not (0.0 <= args.threshold <= 1.0):
            raise ValueError("--threshold must be between 0.0 and 1.0")
        cfg.match.threshold = args.threshold

    # If the user provided a cooldown, validate it and store it
    if args.cooldown is not None:
        if args.cooldown < 0:
            raise ValueError("--cooldown must be non-negative")
        cfg.click.cooldown_seconds = args.cooldown

    # Enable prediction only if the CLI flag is present
    if args.enable_prediction:
        cfg.prediction.enabled = True

    # Override prediction history size if provided; at least 2 points are needed for motion
    if args.prediction_history_size is not None:
        if args.prediction_history_size < 2:
            raise ValueError("--prediction-history-size must be at least 2")
        cfg.prediction.history_size = args.prediction_history_size

    # Override minimum history required before prediction is allowed
    if args.prediction_min_history is not None:
        if args.prediction_min_history < 2:
            raise ValueError("--prediction-min-history must be at least 2")
        cfg.prediction.min_history = args.prediction_min_history

    # Override prediction update threshold if provided
    if args.prediction_update_threshold is not None:
        if not (0.0 <= args.prediction_update_threshold <= 1.0):
            raise ValueError("--prediction-update-threshold must be between 0.0 and 1.0")
        cfg.prediction.update_threshold = args.prediction_update_threshold

    # Override minimum motion threshold if provided
    if args.prediction_min_motion is not None:
        if args.prediction_min_motion < 0:
            raise ValueError("--prediction-min-motion must be non-negative")
        cfg.prediction.min_motion_px = args.prediction_min_motion

    # Override minimum direction consistency if provided
    # Valid range is -1.0 to 1.0 because directional agreement scores naturally live in that interval
    if args.prediction_direction_consistency is not None:
        if not (-1.0 <= args.prediction_direction_consistency <= 1.0):
            raise ValueError("--prediction-direction-consistency must be between -1.0 and 1.0")
        cfg.prediction.min_direction_consistency = args.prediction_direction_consistency

    # Override how far ahead prediction aims
    if args.prediction_scale is not None:
        if args.prediction_scale < 0:
            raise ValueError("--prediction-scale must be non-negative")
        cfg.prediction.prediction_scale = args.prediction_scale

    # Override maximum allowed forward prediction jump
    if args.prediction_max_jump is not None:
        if args.prediction_max_jump < 0:
            raise ValueError("--prediction-max-jump must be non-negative")
        cfg.prediction.max_prediction_px = args.prediction_max_jump

    # Override how many weak/missing frames are allowed before forgetting motion history
    if args.prediction_max_missed_frames is not None:
        if args.prediction_max_missed_frames < 0:
            raise ValueError("--prediction-max-missed-frames must be non-negative")
        cfg.prediction.max_missed_frames = args.prediction_max_missed_frames

    # Copy the debug-window CLI flag into config
    # bool(...) is slightly redundant here because store_true already gives a boolean,
    # but it makes the intention explicit.
    cfg.debug.show_window = bool(args.show_window)

    # Enable saving debug frames only if the CLI flag is present
    if args.save_debug_frames:
        cfg.debug.save_debug_frames = True

    # Return the updated configuration object
    return cfg


def print_startup_summary(cfg: BotConfig) -> None:
    """
    Print the final active startup configuration in a human-readable form.
    """
    print("Starting Duck Hunt bot") 
    print(f"Bridge base URL: {cfg.bridge.base_url}")
    print(f"Templates: {[str(path) for path in cfg.templates.paths]}")
    print(f"Use grayscale: {cfg.templates.use_gray}")
    print(f"Match threshold: {cfg.match.threshold}")
    print(f"Match scales: {list(cfg.match.scales)}") 
    print(f"Click cooldown: {cfg.click.cooldown_seconds}")
    print(f"Prediction enabled: {cfg.prediction.enabled}")
    print(f"Prediction history size: {cfg.prediction.history_size}")
    print(f"Prediction min history: {cfg.prediction.min_history}")
    print(f"Prediction update threshold: {cfg.prediction.update_threshold}")
    print(f"Prediction min motion px: {cfg.prediction.min_motion_px}")
    print(f"Prediction direction consistency: {cfg.prediction.min_direction_consistency}")
    print(f"Prediction scale: {cfg.prediction.prediction_scale}")
    print(f"Prediction max jump px: {cfg.prediction.max_prediction_px}")
    print(f"Prediction max missed frames: {cfg.prediction.max_missed_frames}")
    print(f"Show debug window: {cfg.debug.show_window}")
    print(f"Save debug frames: {cfg.debug.save_debug_frames}")
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
    """
    # Build the command-line parser and read the actual command line
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        # Start from default config values defined in config.py
        cfg = BotConfig()

        # Apply any command-line overrides on top of the defaults
        cfg = apply_cli_overrides(cfg, args)

        # Unless quiet mode is enabled, print the final active settings
        if not args.quiet:
            print_startup_summary(cfg)

        # Create the main controller object and start the bot loop
        controller = DartMasterController(cfg)
        controller.run()

        # Return success exit code
        return 0

    except KeyboardInterrupt:
        # Ctrl+C from the user is treated as a normal manual stop, not a crash
        print("\nStopped by user.")
        return 0

    except Exception as exc:
        # Any other exception is treated as an error
        # Print it to stderr and return non-zero exit code
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    # main() returns an integer exit code; SystemExit passes it to the operating system
    raise SystemExit(main())