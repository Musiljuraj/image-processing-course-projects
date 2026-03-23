import requests

"""
Low-level click communication test. Used during developing.
"""

BRIDGE_BASE = "http://127.0.0.1:8080"


def main() -> None:
    response = requests.post(
        f"{BRIDGE_BASE}/click_local",
        json={"x": 3800, "y": 1700},
        timeout=2,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()