"""
Configuration setting center of the whole project
"""

from dataclasses import dataclass, field  # dataclass = easy data-holding classes, field = custom default creation
from pathlib import Path                  # Path = modern object for filesystem paths
import cv2                               # OpenCV; used here for the template-matching method constant


@dataclass
class BridgeConfig:
    """
    HTTP bridge addresses and timeout.
    This groups all settings related to communication with the Windows bridge.
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
    May use multiple templates to match on, but it slows things down too much.
    """
    # Use default_factory so each TemplateConfig gets its own fresh list of template paths.
    # The currently active default uses the main dartboard template.
    # Another template is shown but commented out.
    paths: list[Path] = field(default_factory=lambda: [
        Path("assets/templates/duck_side_ld.png"),
        #Path("assets/templates/duck_side_lm.png"),
        #Path("assets/templates/duck_side_lu.png"),
        #Path("assets/templates/duck_side_rd.png"),
        #Path("assets/templates/duck_side_rm.png"),
        Path("assets/templates/duck_side_ru.png"),
        #Path("assets/templates/duck_up_ld.png"),
        #Path("assets/templates/duck_up_lm.png"),
        Path("assets/templates/duck_up_lu.png"),
        #Path("assets/templates/duck_up_rd.png"),
        #Path("assets/templates/duck_up_rm.png"),
        Path("assets/templates/duck_up_ru.png"),
    ])

    # Whether frames/templates should be processed in grayscale mode by default
    use_gray: bool = True


@dataclass
class MatchConfig:
    """Matching method, threshold and scales."""
    #method: int = cv2.TM_SQDIFF_NORMED  #alternative method - this one is best
    #method: int = cv2.TM_CCORR_NORMED  #alternative method -
    method: int = cv2.TM_CCOEFF_NORMED  # OpenCV template-matching method used by default
    
    threshold: float = 0.70             # Minimum score required to accept a match

    scales: tuple[float, ...] = (0.50,)  # scaling of templates
    #scales: tuple[float, ...] = (1.00,)  # scaling of templates

    use_multi_match: bool = False       # Whether to use multi-candidate matching logic
    multi_threshold: float = 0.78       # Threshold for multi-match mode
    max_candidates: int = 5             # Maximum number of candidates in multi-match mode


@dataclass
class ClickConfig:
    """
    Cooldown interval - wait interval between two darts
    (so that there will be no two shots on one target).
    """
    cooldown_seconds: float = 0.03      # Wait time between clicks


@dataclass
class PredictionConfig:
    """
    Optional predictive clicking (i.e. basic prediction of movement of a target).
    Keep OFF by default so plain "python3 main.py" preserves the strong static baseline behavior.
    """
    enabled: bool = False               # Prediction is disabled by default

    # Use only a short history; too long reacts badly to direction changes.
    history_size: int = 4               # Maximum number of recent detected centers kept in history
    min_history: int = 3                # Minimum history length required before prediction is allowed

    # Only good enough matches update motion history.
    update_threshold: float = 0.60      # Minimum match score required to trust a point for motion history

    # Small movement is treated as jitter, not true motion.
    min_motion_px: float = 6.0          # Ignore very small movements as noise/jitter

    # Direction must be stable enough across recent steps.
    min_direction_consistency: float = 0.80  # Require recent movement direction to be consistent enough

    # How far ahead to click, in units of recent average motion.
    prediction_scale: float = 0.35      # Conservative forward prediction factor

    # Safety cap for forward jump.
    max_prediction_px: int = 30         # Never allow prediction to jump more than this many pixels

    # Forget history after too many weak/missing frames.
    max_missed_frames: int = 2          # Reset motion history if too many frames are unreliable


@dataclass
class DebugConfig:
    """
    Debug output settings.
    """
    show_window: bool = False           # By default, do not show a debug image window
    save_debug_frames: bool = False     # By default, do not save annotated debug frames
    debug_dir: Path = Path("assets/debug_captures")  # Folder for saved debug captures


@dataclass
class BotConfig:
    """
    Top-level container that groups all configurations together.
    This is the single config object passed around the application.
    """
    bridge: BridgeConfig = field(default_factory=BridgeConfig)          # Bridge communication settings
    templates: TemplateConfig = field(default_factory=TemplateConfig)   # Template file and grayscale settings
    match: MatchConfig = field(default_factory=MatchConfig)             # Template matching settings
    click: ClickConfig = field(default_factory=ClickConfig)             # Click timing settings
    prediction: PredictionConfig = field(default_factory=PredictionConfig)  # Optional predictive-click settings
    debug: DebugConfig = field(default_factory=DebugConfig)             # Debug display/save settings
