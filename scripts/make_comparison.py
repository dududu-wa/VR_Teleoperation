"""
Generate 6 side-by-side comparison images from two model playback sources.
- AMP (model_20000): frames directory (PNG files)
- Interrupt (model_30000): mp4 video (extract frames via ffmpeg)

Randomly picks 6 timestamps and creates comparison images.
"""
import os
import random
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

# --- Paths ---
AMP_FRAMES_DIR = (
    "/home/ubuntu/lzxworkspace/codespace/VR_Teleoperation/logs/r2_amp/"
    "May02_14-34-00_r2v2_amp_version10/"
    "20260504_022636_r2amp_May02_14-34-00_r2v2_amp_version10_ckpt_20000_frames_94pwzn4u"
)
INTERRUPT_VIDEO = (
    "/home/ubuntu/lzxworkspace/codespace/VR_Teleoperation/logs/r2_interrupt/"
    "Apr25_23-26-30_r2v2_ppo_version6/"
    "20260504_050924_r2int_Apr25_23-26-30_r2v2_ppo_version6_ckpt_30000.mp4"
)
OUTPUT_DIR = (
    "/home/ubuntu/lzxworkspace/codespace/VR_Teleoperation/logs/comparison_amp_vs_interrupt"
)

NUM_COMPARISONS = 6
TOTAL_FRAMES = 1501
FPS = 50
SEED = 42


def load_amp_frame(frame_idx):
    """Load a frame from the AMP frames directory."""
    path = os.path.join(AMP_FRAMES_DIR, f"frame_{frame_idx:06d}.png")
    return Image.open(path).convert("RGB")


def extract_interrupt_frame(frame_idx, tmp_dir):
    """Extract a single frame from interrupt mp4 using ffmpeg."""
    out_path = os.path.join(tmp_dir, f"int_{frame_idx:06d}.png")
    # Use frame-accurate seeking with -vf select
    cmd = [
        "ffmpeg", "-y",
        "-i", INTERRUPT_VIDEO,
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-vsync", "vfr",
        "-frames:v", "1",
        "-q:v", "1",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return Image.open(out_path).convert("RGB")


def make_comparison(amp_img, int_img, frame_idx, output_path):
    """Create a side-by-side comparison image with labels."""
    w, h = amp_img.size
    gap = 20
    label_h = 60
    canvas_w = w * 2 + gap
    canvas_h = h + label_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(amp_img, (0, label_h))
    canvas.paste(int_img, (w + gap, label_h))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    time_s = frame_idx / FPS

    # Model labels
    draw.text((w // 2 - 130, 12), "AMP (ckpt 20000)", fill=(30, 80, 180), font=font)
    draw.text((w + gap + w // 2 - 160, 12), "Interrupt (ckpt 30000)", fill=(180, 50, 30), font=font)

    # Timestamp overlay
    ts_text = f"t = {time_s:.2f}s  |  frame {frame_idx}"
    bbox = draw.textbbox((0, 0), ts_text, font=font_sm)
    tw = bbox[2] - bbox[0]
    draw.text(((canvas_w - tw) // 2, canvas_h - 30), ts_text, fill=(60, 60, 60), font=font_sm)

    canvas.save(output_path)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pick 6 random frame indices (avoid first/last 1s for settling)
    random.seed(SEED)
    frame_indices = sorted(random.sample(range(50, TOTAL_FRAMES - 50), NUM_COMPARISONS))
    print(f"Selected frames: {frame_indices}")
    print(f"Times: {[f'{f/FPS:.2f}s' for f in frame_indices]}")
    print()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, fidx in enumerate(frame_indices):
            print(f"[{i+1}/{NUM_COMPARISONS}] Processing frame {fidx} (t={fidx/FPS:.2f}s)...")
            amp_img = load_amp_frame(fidx)
            int_img = extract_interrupt_frame(fidx, tmp_dir)
            out_path = os.path.join(OUTPUT_DIR, f"comparison_{i+1:02d}_frame{fidx:04d}.png")
            make_comparison(amp_img, int_img, fidx, out_path)

    print(f"\nDone! {NUM_COMPARISONS} comparison images saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
