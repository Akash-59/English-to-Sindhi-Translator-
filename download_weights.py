"""
Resilient, resumable downloader for NLLB-200 model weights.
Uses HTTP Range requests to resume interrupted downloads and stream with timeouts.
"""

import os
import sys
import time
import requests

MODEL_URL = "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin"
DEST_PATH = os.path.join("models", "translation", "nllb-200-distilled-600M", "pytorch_model.bin")


def download_with_resume(url: str, dest_path: str, max_retries: int = 10):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    temp_path = dest_path + ".part"

    initial_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

    print(f"Target: {dest_path}")
    print(f"Resuming from byte position: {initial_pos:,} ({initial_pos / (1024*1024):.2f} MB)")

    headers = {}
    if initial_pos > 0:
        headers["Range"] = f"bytes={initial_pos}-"

    retry = 0
    while retry < max_retries:
        try:
            print(f"Initiating stream connection (attempt {retry + 1}/{max_retries})...")
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Check response code
            if response.status_code in (200, 206):
                content_length = response.headers.get("content-length")
                total_size = int(content_length) + initial_pos if content_length else None
                
                mode = "ab" if initial_pos > 0 and response.status_code == 206 else "wb"
                if mode == "wb":
                    initial_pos = 0

                downloaded = initial_pos
                start_time = time.time()
                last_report = start_time

                with open(temp_path, mode) as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunk
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_report >= 5.0:  # Print progress every 5 seconds
                                speed_mb = ((downloaded - initial_pos) / (now - start_time)) / (1024 * 1024) if (now > start_time) else 0
                                pct_str = f"({(downloaded / total_size * 100):.1f}%)" if total_size else ""
                                print(f"Progress: {downloaded / (1024*1024):.1f} MB {pct_str} @ {speed_mb:.2f} MB/s")
                                last_report = now

                # Verify completion
                if total_size and downloaded >= total_size:
                    print(f"Download complete! Final size: {downloaded / (1024*1024):.2f} MB")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(temp_path, dest_path)
                    print(f"Model saved to: {dest_path}")
                    return True
                elif not total_size and downloaded > 2_000_000_000:
                    os.rename(temp_path, dest_path)
                    print(f"Download completed without explicit length header.")
                    return True
            else:
                print(f"HTTP status: {response.status_code}")
                time.sleep(3)

        except (requests.exceptions.RequestException, TimeoutError) as e:
            print(f"Network interruption: {e}. Reconnecting in 3s...")
            time.sleep(3)
            initial_pos = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers["Range"] = f"bytes={initial_pos}-"
            retry += 1

    print("Max retries exceeded.")
    return False


if __name__ == "__main__":
    download_with_resume(MODEL_URL, DEST_PATH)
