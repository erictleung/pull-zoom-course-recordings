import base64
import configparser
from datetime import datetime, timedelta
import os
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional
import requests
from transformers import pipeline

# ================= Configuration Loader =================
CONFIG_FILE = Path(__file__).resolve().parent / "secrets.ini"


def load_zoom_credentials(filepath: Path) -> tuple[str, str, str]:
    """Reads Zoom credentials from a local INI file with environment variable fallback."""
    config = configparser.ConfigParser()

    if filepath.exists():
        config.read(filepath)
        account_id = config.get("zoom", "account_id", fallback="")
        client_id = config.get("zoom", "client_id", fallback="")
        client_secret = config.get("zoom", "client_secret", fallback="")
    else:
        # Fallback to environment variables if file is absent
        account_id = os.getenv("ZOOM_ACCOUNT_ID", "")
        client_id = os.getenv("ZOOM_CLIENT_ID", "")
        client_secret = os.getenv("ZOOM_CLIENT_SECRET", "")

    if not all([account_id, client_id, client_secret]):
        sys.exit(
            f"Error: Missing credentials. Please populate {filepath.name} "
            "with account_id, client_id, and client_secret under the [zoom] section."
        )

    return account_id, client_id, client_secret


ZOOM_ACCOUNT_ID, ZOOM_CLIENT_ID, ZOOM_CLIENT_SECRET = load_zoom_credentials(CONFIG_FILE)

# Lazy-loaded Hugging Face summarizer
_summarizer = None


def get_summarizer():
    global _summarizer
    if _summarizer is None:
        print("\nLoading Hugging Face summarization pipeline (distilbart-cnn-12-6)...")
        _summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    return _summarizer


# ================= Zoom API Client =================
def get_zoom_access_token() -> str:
    """Fetches a Bearer token via Server-to-Server OAuth."""
    url = "https://zoom.us/oauth/token"
    credentials = f"{ZOOM_CLIENT_ID}:{ZOOM_CLIENT_SECRET}"
    b64_creds = base64.b64encode(credentials.encode()).decode()

    headers = {
        "Authorization": f"Basic {b64_creds}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    params = {
        "grant_type": "account_credentials",
        "account_id": ZOOM_ACCOUNT_ID,
    }

    resp = requests.post(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        sys.exit(f"Failed to obtain Zoom access token: {resp.status_code} - {resp.text}")
    return resp.json()["access_token"]


def list_recent_recordings(token: str, days_back: int = 14) -> List[Dict]:
    """Retrieves all cloud recordings from the past N days for the host."""
    url = "https://api.zoom.us/v2/users/me/recordings"
    headers = {"Authorization": f"Bearer {token}"}

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {"from": from_date, "page_size": 30}

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    if resp.status_code != 200:
        sys.exit(f"Error fetching recordings: {resp.status_code} - {resp.text}")

    return resp.json().get("meetings", [])


def clean_vtt(vtt_content: str) -> str:
    """Strips timestamps, cue identifiers, and speaker prefixes from WebVTT."""
    cleaned = re.sub(r"^WEBVTT.*?\n\n", "", vtt_content, flags=re.DOTALL)
    cleaned = re.sub(r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*?\n", "", cleaned)
    cleaned = re.sub(r"^\d+\s*\n", "", cleaned, flags=re.MULTILINE)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return " ".join(lines)


def fetch_transcript(recording_files: List[Dict], token: str) -> Optional[str]:
    """Finds and downloads the TRANSCRIPT/CC file for the selected meeting."""
    transcript_file = next(
        (
            f
            for f in recording_files
            if f.get("file_type") in ("TRANSCRIPT", "CC") or f.get("file_extension") == "VTT"
        ),
        None,
    )

    if not transcript_file:
        return None

    download_url = transcript_file.get("download_url")
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(download_url, headers=headers, timeout=30)

    if resp.status_code == 200:
        return clean_vtt(resp.text)
    return None


# ================= Summarization & Formatting =================
def summarize_transcript(transcript: str, max_chunk_words: int = 700) -> str:
    """Splits transcript into chunks and summarizes each via Hugging Face."""
    summarizer = get_summarizer()
    words = transcript.split()
    chunks = [" ".join(words[i : i + max_chunk_words]) for i in range(0, len(words), max_chunk_words)]

    print(f"Summarizing transcript ({len(chunks)} chunks)...")
    bullets = []
    for chunk in chunks[:5]:
        res = summarizer(chunk, max_length=100, min_length=30, do_sample=False, truncation=True)
        summary_text = res[0]["summary_text"].strip()
        bullets.append(f"• {summary_text}")

    return "\n".join(bullets)


def generate_course_announcement(meeting: Dict, transcript: Optional[str]) -> str:
    topic = meeting.get("topic", "Lecture Session")
    start_time = meeting.get("start_time", "N/A")
    duration = meeting.get("duration", 0)
    share_url = meeting.get("share_url", "N/A")
    passcode = meeting.get("recording_play_passcode", "None")

    if transcript:
        summary = summarize_transcript(transcript)
        preview = transcript[:450] + ("..." if len(transcript) > 450 else "")
    else:
        summary = "• Audio transcript is still processing or was not enabled for this cloud recording."
        preview = "N/A"

    return f"""
==================== COPY BELOW THIS LINE ====================
Hi everyone,

The recording and transcript for "{topic}" are now available.

📌 Access Details:
• Recording Link: {share_url}
• Passcode: {passcode}
• Duration: {duration} minutes
• Date/Time: {start_time}

📝 Key Takeaways & Highlights:
{summary}

📄 Transcript Excerpt:
"{preview}"

Let me know if you run into any trouble accessing the materials!
==================== COPY ABOVE THIS LINE ====================
"""


# ================= Interactive Flow =================
def main():
    token = get_zoom_access_token()
    print("Checking Zoom for recent recordings...")
    recordings = list_recent_recordings(token, days_back=14)

    if not recordings:
        print("No recordings found in the last 14 days.")
        return

    print("\n--- Available Recordings ---")
    for i, m in enumerate(recordings, start=1):
        topic = m.get("topic", "Untitled")
        start = m.get("start_time", "Unknown date")
        dur = m.get("duration", 0)
        has_vtt = any(
            f.get("file_type") in ("TRANSCRIPT", "CC") or f.get("file_extension") == "VTT"
            for f in m.get("recording_files", [])
        )
        status = "✅ Transcript Available" if has_vtt else "⚠️ No Transcript File"
        print(f"[{i}] {start} | {topic} ({dur} mins) [{status}]")

    while True:
        choice = input(f"\nSelect a recording to process (1-{len(recordings)}) or 'q' to quit: ").strip()
        if choice.lower() == "q":
            return
        if choice.isdigit() and 1 <= int(choice) <= len(recordings):
            selected = recordings[int(choice) - 1]
            break
        print("Invalid selection. Please try again.")

    print(f"\nProcessing '{selected.get('topic')}'...")
    transcript = fetch_transcript(selected.get("recording_files", []), token)

    announcement = generate_course_announcement(selected, transcript)

    # Print to console
    print(announcement)

    # Save to disk
    output_filename = "latest_course_message.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(announcement)
    print(f"Saved message to {output_filename}")


if __name__ == "__main__":
    main()
