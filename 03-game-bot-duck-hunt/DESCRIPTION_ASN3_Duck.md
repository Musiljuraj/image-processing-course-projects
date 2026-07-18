PROJECT SUMMARY DUCK HUNT



**Purpose:**

This project is a vision-based auto-click bot for a game target stream. Despite the legacy controller name DartMasterController, the actual templates and startup text show that the current project is configured to detect duck targets in a live ROI image stream, find the best target location by template matching, and send an ROI-local click to an external Windows HTTP bridge.



**Mechanism:**

The program repeatedly downloads the current ROI frame from the bridge, optionally converts it to grayscale, compares the frame with all loaded template images across configured scales by OpenCV template matching, selects the strongest MatchResult, optionally updates a short motion history and predicts a slightly forward click point, checks threshold + cooldown, then sends click coordinates back to the bridge. Primary behavior is direct center clicking; prediction is optional and conservative.



**Call stack** / used modules and functions:

**main.py** is the entry point. build\_arg\_parser() defines CLI options, apply\_cli\_overrides() writes CLI values into BotConfig, print\_startup\_summary() prints the active configuration, and main() **creates DartMasterController(cfg) and calls run()**.

**controller.py** is the runtime core. \_\_init\_\_() creates BridgeFrameSource from capture.py, BridgeClickClient from click\_client.py, **TemplateMatcher from matcher.py**, and loads templates by load\_templates() from templates.py. **run()** performs startup checks and then **loops forever through \_run\_one\_iteration()**.

**\_run\_one\_iteration()** calls: frame\_source.grab\_frame() -> prepare\_frame() from preprocessing.py -> matcher.match\_best() -> \_update\_motion\_history() -> \_choose\_click\_point() -> \_predict\_click\_point() and \_clamp\_point() when prediction is enabled -> \_should\_click() -> click\_client.click\_center() if allowed -> \_save\_debug\_frame() if debug saving is enabled.

**capture.py** provides BridgeFrameSource; it calls GET /health, GET /config, and GET /frame.jpg and decodes JPEG bytes into an OpenCV image.

**click\_client.py** provides BridgeClickClient; it sends POST /click\_local with ROI-local integer coordinates.

templates.py loads template image files from disk into Template objects.

**matcher.py** defines MatchResult and TemplateMatcher.match\_best(); it tries every template and every configured scale, runs cv2.matchTemplate, computes rectangle + center, and returns the best overall match.

**preprocessing.py** provides prepare\_frame(); it converts BGR to grayscale when enabled.

**debug\_view.py** only draws annotations for saved debug images; it does not influence detection or clicking decisions.



**Required input** / conditions:

1\. A **running HTTP bridge** at the configured base URL (default: http://127.0.0.1:8080) that serves at least /health, /config, /frame.jpg, and /click\_local.

2\. Template PNG files stored on disk; the current default config expects:

&#x20;  assets/templates/duck\_side\_ld.png

&#x20;  assets/templates/duck\_side\_ru.png

&#x20;  assets/templates/duck\_up\_lu.png

&#x20;  assets/templates/duck\_up\_ru.png

3\. Python dependencies installed on the bot side: opencv-python, numpy, requests.

4\. Correct package layout for imports as written: main.py imports bot.\* modules, so config.py, controller.py, capture.py, click\_client.py, matcher.py, preprocessing.py, templates.py, and debug\_view.py should be placed inside a bot/ package, unless imports are edited.

5\. The bridge must provide ROI frames already cropped to the game region; this project itself does not capture the desktop directly.



**WINDOWS BRIDGE STARTUP PROCEDURE:**

Main bridge file: screen\_bridge\_duck.py



&#x20; 1. Open Google Chrome.

&#x20; 2. Put Chrome into the fixed layout - right upper quarter of a monitor.

&#x20; 3. Open the Duck Hunt page: **https://duckhuntjs.com/**

&#x20; 4. In the Windows bridge folder activate Windows **venv: .venv\\Scripts\\Activate.ps1**

&#x20; 5. Run: **python screen\_bridge\_duck.py**

&#x20; 6. Keep that PowerShell window open.

Current Windows-side ROI (source of truth in bridge):

&#x20; left=1900, top=300, width=1900, height=650





**Output:**

The main output is not a report file but an action: ROI-local click requests sent to the bridge, which should execute the real mouse click externally. Secondary output is terminal logging for each iteration: CLICK / no click / no match, template name, score, scale, matched size, detected center, final click point, prediction flag, and bridge response. Optional file output is a sequence of annotated PNG debug images saved into assets/debug\_captures/ as controller\_debug\_00001.png, controller\_debug\_00002.png, etc. Note: --show-window exists in CLI/config, but in the uploaded runtime path there is no actual on-screen display call; practical debug output is saved frames, not a live window.



**Functional run commands:**

Basic run:

python main.py



Explicit run with key arguments:

python main.py --bridge-base-url http://127.0.0.1:8080 --threshold 0.70 --cooldown 0.03



Run with debug frame saving:

python main.py --bridge-base-url http://127.0.0.1:8080 --threshold 0.70 --cooldown 0.03 --save-debug-frames



Run with predictive clicking enabled:

python main.py --bridge-base-url http://127.0.0.1:8080 --threshold 0.70 --cooldown 0.03 --enable-prediction --prediction-history-size 4 --prediction-min-history 3 --prediction-update-threshold 0.60 --prediction-min-motion 6 --prediction-direction-consistency 0.80 --prediction-scale 0.35 --prediction-max-jump 30 --prediction-max-missed-frames 2



Show all CLI options:

python main.py --help



**Brief run procedure:**

Before running, check that: the bridge is already running and reachable at the chosen URL; /frame.jpg returns the live ROI image; all required template PNG files exist in assets/templates/; Python has cv2, numpy, and requests installed; and the files are arranged so that bot.\* imports work. Then run one of the commands above. After startup, the bot continuously fetches ROI frames, preprocesses them, finds the strongest template match, optionally predicts slight forward motion, clicks when score >= threshold and cooldown allows it, prints loop status to the terminal, and optionally writes annotated debug PNGs into assets/debug\_captures/.

