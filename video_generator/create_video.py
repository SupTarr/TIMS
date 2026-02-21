import requests
import cv2
import os
import numpy as np
import time
import shutil

# --- Configuration ---
CAMERA_ID = "1630"
DURATION_MINUTES = 5
FPS = 2
# Get from cookies in https://cpudapp.bangkok.go.th/bmatraffic/index.aspx
ASP_NET_SESSION_ID = "fofgxxgskj3myzkqcbejm3rh"
# --- End Configuration ---

if not os.path.exists("traffic_images"):
    os.makedirs("traffic_images")

base_url = "https://cpudapp.bangkok.go.th/bmatraffic/show.aspx"

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Host": "cpudapp.bangkok.go.th",
    "Referer": f"https://cpudapp.bangkok.go.th/bmatraffic/PlayVideo.aspx?ID={CAMERA_ID}",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "same-origin",
    "sec-ch-ua": '"Chromium";v="141", "Not(A:Brand";v="8"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"macOS"',
}

session = requests.Session()
session.cookies.set(
    "ASP.NET_SessionId", ASP_NET_SESSION_ID, domain="cpudapp.bangkok.go.th", path="/"
)

print(f"Initializing session for camera {CAMERA_ID}...")
response = session.get(
    f"https://cpudapp.bangkok.go.th/bmatraffic/PlayVideo.aspx?ID={CAMERA_ID}",
    headers={
        "User-Agent": headers["User-Agent"],
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
    timeout=10,
)

print(f"Session cookies: {session.cookies.get_dict()}")
session.headers.update(headers)

image_files = []
count = 0

max_images = DURATION_MINUTES * 60 * FPS
capture_interval = 1 / FPS

print(f"\nStarting image download (max {max_images} images)...\n")

start_time = time.time()
while count < max_images:
    count += 1
    current_time = int(time.time() * 1000)
    image_url = f"{base_url}?image={CAMERA_ID}&&time={current_time}"
    try:
        response = session.get(image_url, timeout=10)
        if response.status_code != 200:
            print(f"[{count}/{max_images}] ERROR: Could not read downloaded image")

        content_type = response.headers.get("Content-Type", "unknown")
        print(
            f"[{count}/{max_images}] Content-Type: {content_type}, Size: {len(response.content)} bytes"
        )

        image_filename = f"traffic_images/bmatraffic_{CAMERA_ID}_{count:04d}.jpg"
        with open(image_filename, "wb") as f:
            f.write(response.content)

        img_check = cv2.imread(image_filename)
        if img_check is None:
            print(
                f"[{count}/{max_images}] Failed to download image. Status code: {response.status_code}"
            )

        mean_val = img_check.mean()
        if mean_val > 250:
            print(
                f"[{count}/{max_images}] WARNING: Image appears to be blank (mean={mean_val:.1f})"
            )
        else:
            image_files.append(image_filename)
            print(
                f"[{count}/{max_images}] Downloaded {image_filename} successfully (mean={mean_val:.1f})"
            )

    except Exception as e:
        print(f"[{count}/{max_images}] Error downloading image: {e}")

    time.sleep(capture_interval)

end_time = time.time()
print(f"\nDownload complete! Total valid images: {len(image_files)}")

if image_files:
    first_image = cv2.imread(image_files[0])
    if first_image is not None:
        height, width, layers = first_image.shape

        start_time_str = time.strftime("%Y%m%dT%H%M%S", time.localtime(start_time))
        end_time_str = time.strftime("%Y%m%dT%H%M%S", time.localtime(end_time))
        video_filename = f"bmatraffic_{CAMERA_ID}_{start_time_str}_{end_time_str}.avi"

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter(video_filename, fourcc, FPS, (width, height))
        print(f"\nCreating video '{video_filename}' at {FPS} FPS...")
        for image_file in image_files:
            img = cv2.imread(image_file)
            if img is not None:
                video.write(img)

        video.release()
        print(f"Video '{video_filename}' created successfully!")
        print(f"Duration: {len(image_files)} seconds")

    else:
        print("Could not read the first image, cannot create video.")
else:
    print("No images were downloaded, cannot create video.")

try:
    if os.path.exists("traffic_images"):
        print("\nCleaning up temporary image files...")
        shutil.rmtree("traffic_images")
        print("Directory 'traffic_images' removed successfully.")
except Exception as e:
    print(f"Error during cleanup: {e}")
