import os
import yt_dlp

from pydub import AudioSegment

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Download YouTube Audio
# ─────────────────────────────────────────────
def download_youtube_audio(url: str) -> str:

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,

        # Better compatibility
        "noplaylist": True,
        "quiet": True,
        "nocheckcertificate": True,
        "ignoreerrors": False,

        # Prevent bot blocking
        "extractor_args": {
            "youtube": {
                "player_client": ["android"]
            }
        },

        # Convert to wav automatically
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            info = ydl.extract_info(url, download=True)

            downloaded_file = ydl.prepare_filename(info)

            base = os.path.splitext(downloaded_file)[0]
            wav_file = base + ".wav"

            if not os.path.exists(wav_file):
                raise Exception("WAV conversion failed.")

            return wav_file

    except Exception as e:
        raise Exception(f"YouTube download failed: {str(e)}")


# ─────────────────────────────────────────────
# Convert Local File To WAV
# ─────────────────────────────────────────────
def convert_to_wav(input_path: str) -> str:

    try:
        output_path = (
            os.path.splitext(input_path)[0]
            + "_converted.wav"
        )

        audio = AudioSegment.from_file(input_path)

        # Whisper-friendly format
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        audio.export(output_path, format="wav")

        return output_path

    except Exception as e:
        raise Exception(f"Audio conversion failed: {str(e)}")


# ─────────────────────────────────────────────
# Chunk Audio
# ─────────────────────────────────────────────
def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    try:
        audio = AudioSegment.from_wav(wav_path)

        chunk_ms = chunk_minutes * 60 * 1000

        chunks = []

        for i, start in enumerate(
            range(0, len(audio), chunk_ms)
        ):

            chunk = audio[start:start + chunk_ms]

            chunk_path = (
                f"{wav_path}_chunk_{i}.wav"
            )

            chunk.export(chunk_path, format="wav")

            chunks.append(chunk_path)

        return chunks

    except Exception as e:
        raise Exception(f"Audio chunking failed: {str(e)}")


# ─────────────────────────────────────────────
# Main Processing Pipeline
# ─────────────────────────────────────────────
def process_input(source: str) -> list:

    try:

        # YouTube URL
        if source.startswith("http://") or source.startswith("https://"):

            print("Detected YouTube URL")

            wav_path = download_youtube_audio(source)

        # Local file
        else:

            print("Detected local file")

            wav_path = convert_to_wav(source)

        print("Chunking audio...")

        chunks = chunk_audio(wav_path)

        print(
            f"Audio ready — {len(chunks)} chunk(s) created."
        )

        return chunks

    except Exception as e:
        raise Exception(
            f"Audio processing failed: {str(e)}"
        )
        