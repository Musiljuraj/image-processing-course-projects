**PROJECT SUMMARY — DART MASTER BOT**



Purpose:

This project is a vision-based auto-click bot for the Dart Master browser game. Its goal is to detect the dartboard target in a live ROI screenshot stream, choose the best matching location by template matching, and send an ROI-local click to an external Windows bridge. Base behavior is direct-center clicking; optional prediction slightly shifts the click forward using short recent motion history.



How it works:

**main.py** is only the startup/orchestration layer. It parses CLI arguments, creates BotConfig, applies overrides, prints startup config, creates DartMasterController, and starts the loop.

**controller.py** is the real runtime core: it checks bridge health, loads template images once, then repeatedly grabs the current ROI frame from the bridge, preprocesses it, matches all templates, picks the strongest MatchResult, optionally updates motion history and predicts a forward click point, checks score threshold + click cooldown, sends click\_local to the bridge, and optionally saves annotated debug PNGs.

**capture.py** fetches /frame.jpg and decodes JPEG to OpenCV BGR image. preprocessing.py converts the frame to grayscale if configured.

**templates.py** loads template PNGs from disk into Template objects.

**matcher.py** runs cv2.matchTemplate for each template and returns the best normalized result (template name, score, rectangle, center).

**click\_client.py** POSTs ROI-local click coordinates to the Windows bridge.

**debug\_view.py** only visualizes the result; it does not affect decisions.



**Call stack** / used modules and functions:

**main.py** -> build\_arg\_parser() -> apply\_cli\_overrides() -> print\_startup\_summary() -> main()

\-> **DartMasterController(cfg)**

**controller.py** -> \_\_init\_\_() creates:

&#x20; BridgeFrameSource(cfg.bridge) from capture.py

&#x20; BridgeClickClient(cfg.bridge) from click\_client.py

&#x20; TemplateMatcher(cfg.match.method) from matcher.py

&#x20; load\_templates(cfg.templates.paths, use\_gray=...) from templates.py

**controller.py -> run() -> \_run\_one\_iteration()**

**\_run\_one\_iteration():**

&#x20; frame\_source.grab\_frame()

&#x20; prepare\_frame(frame\_bgr, use\_gray=...)

&#x20; **matcher.match\_best(work\_frame, templates)**

&#x20; \_update\_motion\_history(best\_match)

&#x20; \_choose\_click\_point(best\_match, frame\_shape)

&#x20; \_should\_click(best\_match, click\_point)

&#x20; click\_client.click\_center(click\_point)   \[only if allowed]

&#x20; \_save\_debug\_frame(...)                   \[only if debug saving enabled]

Prediction helpers in controller.py: \_predict\_click\_point(), \_clamp\_point().



**Required inputs:**

1\. **Live Windows bridge** server running at the configured URL (default http://127.0.0.1:8080) with endpoints:

&#x20;  GET /health

&#x20;  GET /config

&#x20;  GET /frame.jpg

&#x20;  POST /click\_local

2\. **Template image(s) on disk**; default active template is:

&#x20;  assets/templates/dartboard\_main.png

&#x20;  (optional alternative template also exists in notes/config: assets/templates/dartboard\_alt\_01.png)

3\. Correct project/package layout: main.py imports bot.\* modules, so support files must be inside a bot/ package directory when run as written.

4\. Python packages:

&#x20;  WSL / bot side: opencv-python, numpy, requests

&#x20;  Windows bridge side (.venv): flask, pillow, pynput



**What the project outputs:**

Primary output is action: **ROI-local click requests sent to the Windows bridge**, which then performs the real mouse click on Windows. Secondary output is terminal logging for each loop iteration (“CLICK”, “no click”, “no match”, score, template, detected center, final click point, prediction flag, bridge response). Optional file output: annotated debug frames saved to assets/debug\_captures/controller\_debug\_00001.png, controller\_debug\_00002.png, ... when --save-debug-frames is enabled.



**WINDOWS BRIDGE STARTUP PROCEDURE:**

Windows bridge folder:

&#x20; C:\\Users\\JM\\zao-camera-stream

Main bridge file:

&#x20; screen\_bridge.py

Used by WSL project:

&#x20; \~/3rls/ZAO/DU/03\_dart\_master\_bot

Bridge purpose:

&#x20; captures the game ROI from the Windows desktop and executes mouse clicks requested by the WSL bot

Startup steps on Windows:

&#x20; **1. Open Google Chrome.**

&#x20; **2. Put Chrome into the fixed layout used for this project.**

&#x20; **3. Open the Dart Master page: https://www.marketjs.com/item/dart-master/**

&#x20; **4. In C:\\Users\\JM\\zao-camera-stream activate Windows venv:**

&#x20;    **.venv\\Scripts\\Activate.ps1**

&#x20; **5. Run:**

&#x20;    **python screen\_bridge.py**

&#x20; 6. Keep that PowerShell window open.

Current Windows-side ROI (source of truth in bridge):

&#x20; left=2042, top=979, width=1636, height=926

Important:

&#x20; ROI is owned by screen\_bridge.py, not by the WSL bot. If ROI changes, edit screen\_bridge.py and restart the bridge. The bridge must be running before the bot starts.



**Functional run commands:**



Basic run:

&#x20; **python main.py**



Explicit default-like run:

&#x20; python main.py --bridge-base-url http://127.0.0.1:8080 --threshold 0.72 --cooldown 0.5



Run with saved debug images:

&#x20; python main.py --save-debug-frames



Run with predictive clicking enabled:

&#x20; python main.py --enable-prediction --prediction-history-size 4 --prediction-min-history 3 --prediction-update-threshold 0.60 --prediction-min-motion 6 --prediction-direction-consistency 0.80 --prediction-scale 0.35 --prediction-max-jump 30 --prediction-max-missed-frames 2



Show CLI help:

&#x20; python main.py --help



**Complete startup / run procedure:**

1\. On Windows, open Chrome, snap it to the right half of the primary monitor, keep zoom at 100%, open the Dart Master page, and start the game manually.

2\. In C:\\Users\\JM\\zao-camera-stream activate .venv and run:

&#x20;  python screen\_bridge.py

3\. Verify the bridge is alive at http://127.0.0.1:8080 and that its ROI still matches the game area.

4\. On the bot side, ensure project layout matches imports (support modules under bot/), install opencv-python + numpy + requests, and ensure assets/templates/dartboard\_main.png exists.

5\. Run:

&#x20;  python main.py

&#x20;  or one of the explicit variants above.

6\. During runtime the bot repeatedly reads /frame.jpg, finds the best template match, clicks when score >= threshold and cooldown allows it, and optionally stores annotated debug PNGs in assets/debug\_captures/.

