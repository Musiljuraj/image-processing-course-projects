"""
Central configuration definition for the whole project.

This module contains the complete configuration tree used by the bot. The rest of the
implementation assumes that there is one shared top-level BotConfig object, and that this
object groups together all settings needed by the runtime system.

In practice, this module defines the "static structure" of the bot before any work starts.
main.py creates BotConfig as the default startup state, then optionally overrides selected
fields from the command line, and finally passes that single config object into the controller.

Because of that role, config.py is not responsible for runtime behavior itself.
Instead, it describes the parameters that shape runtime behavior in other modules:

- capture.py uses bridge settings
- click_client.py uses bridge settings
- templates.py and preprocessing.py depend on template-related settings
- matcher.py depends on matching settings
- controller.py depends on nearly all sections because it coordinates the full runtime flow
- debug_view.py is influenced indirectly through debug and threshold-related settings

The dataclass-based structure is important here:
each config section is its own small typed object, and BotConfig groups them together into one
hierarchical configuration model that is easy to create, pass around, inspect, and override.
"""

from dataclasses import dataclass, field  # dataclass = easy data-holding classes, field = custom default creation
from pathlib import Path                  # Path = modern object for filesystem paths
import cv2                               # OpenCV; used here for the template-matching method constant


@dataclass
class BridgeConfig:
    """
    HTTP bridge addresses and timeout.

    This config section contains everything needed for communication with the Windows bridge.
    The bridge is the external component that provides screenshots and performs actual clicks.

    Other modules do not hardcode bridge URLs or endpoints by themselves.
    Instead, they receive BridgeConfig and use its values when constructing requests.

    This keeps all bridge communication settings centralized in one place.
    """
    base_url: str = "http://127.0.0.1:8080"   # Base URL of the bridge server
    health_endpoint: str = "/health"          # Endpoint for bridge health check
    config_endpoint: str = "/config"          # Endpoint for reading bridge-side configuration
    frame_endpoint: str = "/frame.jpg"        # Endpoint for downloading the current ROI frame
    click_endpoint: str = "/click_local"      # Endpoint for sending ROI-local click commands
    timeout_seconds: float = 2.0              # HTTP timeout for requests to the bridge


@dataclass
class TemplateConfig:
    """
    Template paths and grayscale mode.

    This section defines which template image files should be loaded and whether the project
    should operate in grayscale mode by default.

    The template list represents the visual references that matcher.py will try to find inside
    the incoming bridge frames. Even though the system supports multiple templates, the comment
    in the original implementation already reflects an important practical tradeoff:
    more templates increase flexibility, but also increase matching cost.

    The grayscale flag is part of the project's consistent frame/template preparation path:
    templates.py uses it when loading template images, and preprocessing.py uses the same choice
    when preparing live frames, so both sides stay in the same image format.
    """
    # Use default_factory so each TemplateConfig gets its own fresh list of template paths.
    # This avoids sharing one mutable list between multiple config instances.
    #
    # The currently active default uses the main dartboard template.
    # Another template is shown but commented out, which makes the intended extension path visible
    # without changing the current active behavior.
    paths: list[Path] = field(default_factory=lambda: [
        Path("assets/templates/dartboard_main.png"),
        #Path("assets/templates/dartboard_alt_01.png"),
    ])

    # Whether frames/templates should be processed in grayscale mode by default.
    # This setting must stay aligned across template loading and live frame preparation.
    use_gray: bool = True


@dataclass
class MatchConfig:
    """
    Matching method and threshold.

    This section controls how template matching is performed and how strong a result must be
    before the controller treats it as click-worthy.

    The selected OpenCV method influences how score maps are interpreted in matcher.py.
    In particular, the SQDIFF family behaves differently from correlation-based methods,
    so matcher.py contains normalization logic that converts the chosen method into one common
    "higher score is better" interpretation.

    The remaining fields describe both the main single-best-match path and an optional
    multi-candidate matching path, even though the current controller logic appears focused
    on the single-best-match baseline.
    """
    # OpenCV template matching method used by default.
    # The original note indicates this method was judged best for the current project behavior.
    method: int = cv2.TM_SQDIFF_NORMED  #alternative method - this one is best
    #method: int = cv2.TM_CCORR_NORMED  #alternative method -
    #method: int = cv2.TM_CCOEFF_NORMED  # OpenCV template-matching method used by default

    # Minimum score required to accept a match for clicking logic.
    # matcher.py always returns the strongest candidate it found; controller.py later compares that
    # score to this threshold to decide whether a click is allowed.
    threshold: float = 0.72             # Minimum score required to accept a match

    # Optional settings for a broader multi-candidate matching mode.
    # These are part of the configuration structure even if the current flow mainly uses match_best().
    use_multi_match: bool = False       # Whether to use multi-candidate matching logic
    multi_threshold: float = 0.78       # Threshold for multi-match mode
    max_candidates: int = 5             # Maximum number of candidates in multi-match mode


@dataclass
class ClickConfig:
    """
    Click pacing settings.

    The core idea here is that successful visual detection should not automatically mean
    unlimited rapid clicking. The controller uses this cooldown interval to prevent repeated
    clicks from firing too close together.

    In the current runtime flow, controller.py uses this value in two ways:
    - to decide whether enough time has passed since the previous click
    - to sleep after a click so the loop naturally waits between shots

    That makes this setting part of both correctness and pacing.
    """
    # Wait interval between two darts so there will be no two shots on one target.
    cooldown_seconds: float = 0.5       # Wait time between clicks


@dataclass
class PredictionConfig:
    """
    Optional predictive clicking configuration.

    This section contains all parameters related to conservative short-history motion prediction.
    The design here is intentionally safety-oriented: prediction exists as an optional enhancement,
    not as the baseline behavior.

    The original comment captures the intended philosophy very clearly:
    prediction stays OFF by default so plain startup preserves the stronger static baseline.
    When enabled, controller.py uses these settings to decide:

    - whether enough center history exists
    - whether recent movement is real motion or just jitter
    - whether recent movement direction is stable enough
    - how far ahead to shift the click point
    - when old motion history should be discarded

    So this config block effectively defines the "safety envelope" for predictive clicking.
    """
    # Prediction is disabled by default so the system starts in the stable direct-center mode.
    enabled: bool = False               # Prediction is disabled by default

    # Use only a short history; too long reacts badly to direction changes.
    # These values define the size of the recent-center memory and the minimum amount of evidence
    # needed before a prediction is even considered.
    history_size: int = 4               # Maximum number of recent detected centers kept in history
    min_history: int = 3                # Minimum history length required before prediction is allowed

    # Only good enough matches update motion history.
    # This prevents unreliable detections from polluting the motion model.
    update_threshold: float = 0.60      # Minimum match score required to trust a point for motion history

    # Small movement is treated as jitter, not true motion.
    # This avoids generating predictions from tiny unstable pixel changes.
    min_motion_px: float = 6.0          # Ignore very small movements as noise/jitter

    # Direction must be stable enough across recent steps.
    # controller.py computes an average direction-consistency score and compares it to this threshold.
    min_direction_consistency: float = 0.80  # Require recent movement direction to be consistent enough

    # How far ahead to click, in units of recent average motion.
    # This is a deliberately conservative forward factor rather than a full extrapolation.
    prediction_scale: float = 0.35      # Conservative forward prediction factor

    # Safety cap for forward jump.
    # Even if the recent motion estimate suggests a larger move, the final forward shift is clamped.
    max_prediction_px: int = 30         # Never allow prediction to jump more than this many pixels

    # Forget history after too many weak/missing frames.
    # This prevents stale movement from surviving through unreliable detection gaps.
    max_missed_frames: int = 2          # Reset motion history if too many frames are unreliable


@dataclass
class DebugConfig:
    """
    Debug output settings.

    This section controls non-essential runtime outputs used for inspection and diagnostics.
    The main project behavior does not depend on these settings, but they shape whether
    supporting visual artifacts are displayed or saved.

    controller.py uses these values when deciding whether to save annotated frames,
    and where those saved frames should be written.
    """
    show_window: bool = False           # By default, do not show a debug image window
    save_debug_frames: bool = False     # By default, do not save annotated debug frames
    debug_dir: Path = Path("assets/debug_captures")  # Folder for saved debug captures


@dataclass
class BotConfig:
    """
    Top-level container that groups all configuration sections together.

    This is the single shared config object used across the entire project.
    main.py creates it once, applies CLI overrides to it, and then passes it into
    DartMasterController, which distributes the relevant sub-configs to lower-level components.

    Structurally, BotConfig is the root of the configuration tree:

    - bridge     -> HTTP communication with the Windows bridge
    - templates  -> template file paths and grayscale selection
    - match      -> template matching behavior
    - click      -> click pacing
    - prediction -> optional conservative motion prediction
    - debug      -> optional debugging artifacts

    Using one top-level config object keeps the project consistent:
    every module reads from the same runtime settings rather than managing its own independent state.
    """
    # Each section uses default_factory so every BotConfig instance receives fresh nested config objects.
    # This avoids accidental shared mutable state between separate BotConfig instances.
    bridge: BridgeConfig = field(default_factory=BridgeConfig)          # Bridge communication settings
    templates: TemplateConfig = field(default_factory=TemplateConfig)   # Template file and grayscale settings
    match: MatchConfig = field(default_factory=MatchConfig)             # Template matching settings
    click: ClickConfig = field(default_factory=ClickConfig)             # Click timing settings
    prediction: PredictionConfig = field(default_factory=PredictionConfig)  # Optional predictive-click settings
    debug: DebugConfig = field(default_factory=DebugConfig)             # Debug display/save settings