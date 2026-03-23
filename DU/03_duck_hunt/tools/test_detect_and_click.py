import cv2

from bot.capture import BridgeFrameSource
from bot.click_client import BridgeClickClient
from bot.config import BotConfig
from bot.debug_view import draw_match
from bot.matcher import TemplateMatcher
from bot.preprocessing import prepare_frame
from bot.templates import load_templates

"""
Multi-scale single-cycle integration test.
It performs one end-to-end pass: fetch frame, preprocess, load templates,
match target across configured templates and scales, threshold check,
send click if passed, save annotated debug image.
Used during development. 
"""

def main() -> None:
    cfg = BotConfig()

    out_path = cfg.debug.debug_dir / "detect_and_click_debug.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_source = BridgeFrameSource(cfg.bridge)
    click_client = BridgeClickClient(cfg.bridge)

    health = frame_source.check_health()
    print("Bridge health:", health)

    frame_bgr = frame_source.grab_frame()
    work_frame = prepare_frame(frame_bgr, use_gray=cfg.templates.use_gray)

    templates = load_templates(
        cfg.templates.paths,
        use_gray=cfg.templates.use_gray,
    )
    print("Loaded templates:", [template.name for template in templates])

    print("Configured scales:", list(cfg.match.scales))  
    matcher = TemplateMatcher(cfg.match.method, cfg.match.scales) 
    best_match = matcher.match_best(work_frame, templates)

    clicked = False
    click_response = None
    click_point = None  

    if best_match is None:
        print("No match result was produced.")
    else:
        print(f"Best template: {best_match.template_name}")
        print(f"Score: {best_match.score:.4f}")
        print(f"Winning scale: {best_match.scale:.2f}")  
        print(f"Matched width: {best_match.matched_width}") 
        print(f"Matched height: {best_match.matched_height}")  
        print(f"Top-left: {best_match.top_left}")
        print(f"Bottom-right: {best_match.bottom_right}")
        print(f"Center: {best_match.center}")

        threshold_pass = best_match.score >= cfg.match.threshold
        print(f"Threshold pass: {threshold_pass}")

        if threshold_pass:
            click_point = best_match.center 
            click_response = click_client.click_center(click_point)  
            clicked = True
            print("Click point:", click_point)  
            print("Click response:", click_response)
        else:
            print("Match score is below threshold, no click sent.")

    annotated = draw_match(
        frame_bgr=frame_bgr,
        match=best_match,
        threshold=cfg.match.threshold,
        click_point=click_point, 
        prediction_used=False, 
    )

    status_text = f"clicked={clicked}"
    cv2.putText(
        annotated,
        status_text,
        (10, 145),  
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    ok = cv2.imwrite(str(out_path), annotated)
    if not ok:
        raise RuntimeError(f"Could not save debug image to {out_path}")

    print(f"Saved debug image: {out_path}")

    if cfg.debug.show_window:
        cv2.imshow("detect_and_click_debug", annotated)
        print("Press any key in the debug window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()