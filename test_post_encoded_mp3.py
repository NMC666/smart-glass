from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


def ensure_wav(sample_wav: Path, fallback_source: Path) -> Path:
    if sample_wav.exists():
        return sample_wav

    if not fallback_source.exists():
        raise FileNotFoundError(
            f"No sample wav found at {sample_wav} and no fallback source at {fallback_source}"
        )

    sample_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(fallback_source),
        str(sample_wav),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Failed to convert m4a to wav with ffmpeg. "
            f"stderr: {completed.stderr.strip()}"
        )

    return sample_wav


def post_wav(api_url: str, audio_path: Path) -> tuple[int, str]:
    audio_bytes = audio_path.read_bytes()
    req = Request(
        api_url,
        data=audio_bytes,
        headers={
            "Content-Type": "audio/wav",
            "Accept": "application/json",
            "X-Filename": audio_path.name,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )

    try:
        with urlopen(req, timeout=120) as resp:
            status = resp.status
            text = resp.read().decode("utf-8", errors="replace")
            return status, text
    except HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return exc.code, text
    except URLError as exc:
        raise RuntimeError(f"Failed to reach server: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post a WAV sample to voice Flask /transcribe",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:5052/audio",
        help="Voice HTTP endpoint (main.py default is 5051; voice CLI server is 5052)",
    )
    parser.add_argument(
        "--sample-wav",
        default="raw_data/mouse.wav",
        help="Path to sample WAV (will be generated if missing)",
    )
    parser.add_argument(
        "--fallback-source",
        default="raw_data/mouse.mp3",
        help="Fallback audio source for wav conversion (mp3/m4a/wav supported by ffmpeg)",
    )
    args = parser.parse_args()

    sample_wav = ensure_wav(Path(args.sample_wav), Path(args.fallback_source))
    print(f"Using WAV sample: {sample_wav}")
    print(f"Posting to: {args.api_url}")
    status, response = post_wav(args.api_url, sample_wav)

    print(f"POST status: {status}")
    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
