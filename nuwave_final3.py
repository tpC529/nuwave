"""
4nuwavegradio.py  –  NuWave Video Generator
============================================
Upload audio + image → Generate Video → download MP4.

Key change: waveform rendered entirely inside FFmpeg (showwaves filter).
No Python frame loop. A 1h15m audio file renders in ~1-2 min instead of hours.

Pipeline:
  1. Load audio, derive color theme from image
  2. LTX-Video AI animates the image (looped to audio length) — if available
     OR static image — if not
  3. FFmpeg renders waveform + composites background + muxes audio in one pass

Install:
    pip install gradio librosa pydub imageio-ffmpeg pillow numpy psutil

AI video (optional):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install diffusers transformers accelerate sentencepiece

Run:  python 4nuwavegradio.py  →  http://localhost:7860
"""

import os, re, json, subprocess, tempfile, traceback, math
import numpy as np
import psutil
from PIL import Image

# ── Optional audio libs ────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

# ── FFmpeg binary ──────────────────────────────────────────────────────────────
try:
    import imageio_ffmpeg
    _ffp = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ['PATH'] = os.path.dirname(_ffp) + os.pathsep + os.environ.get('PATH', '')
    if PYDUB_OK:
        AudioSegment.converter = _ffp
        AudioSegment.ffmpeg    = _ffp
    FFMPEG_CMD = _ffp
except Exception as _fe:
    print(f'Warning: bundled FFmpeg not found – {_fe}')
    FFMPEG_CMD = 'ffmpeg'

# ── LTX-Video ─────────────────────────────────────────────────────────────────
LTX_AVAILABLE = False
try:
    import torch
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video
    # Only mark as available if CUDA is actually usable
    if torch.cuda.is_available():
        LTX_AVAILABLE = True
    else:
        print('LTX-Video: torch imported but CUDA not available – AI mode disabled')
except ImportError:
    pass

WORK_DIR = os.path.join(tempfile.gettempdir(), 'nuwave_work')
os.makedirs(WORK_DIR, exist_ok=True)

# ── NVENC detection ────────────────────────────────────────────────────────────
# Test whether the RTX GPU's hardware encoder is available.
# If yes: encodes 5-10x faster than CPU libx264 (245s → ~30s for 1h15m).
# If no: silently falls back to libx264 so the app still works on any machine.
def _detect_nvenc() -> bool:
    try:
        r = subprocess.run(
            [FFMPEG_CMD, '-f', 'lavfi', '-i', 'nullsrc=s=64x64:d=1',
             '-c:v', 'h264_nvenc', '-f', 'null', '-'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        return r.returncode == 0
    except Exception:
        return False

NVENC_AVAILABLE = _detect_nvenc()
if NVENC_AVAILABLE:
    print('✓ NVENC detected — using GPU hardware encoding (h264_nvenc)')
    # NVENC quality flags: -cq (constant quality) replaces -crf, -preset p4 = good balance
    _VC   = ['h264_nvenc', '-preset', 'p4', '-cq', '22']
    # For still-image sources NVENC doesn't support -tune stillimage — omit it
    _VC_STILL = ['h264_nvenc', '-preset', 'p4', '-cq', '22']
else:
    print('  NVENC not available — using CPU encoding (libx264)')
    _VC       = ['libx264', '-preset', 'fast', '-crf', '22']
    _VC_STILL = ['libx264', '-preset', 'fast', '-crf', '22', '-tune', 'stillimage']

VID_W    = 704
VID_H    = 480
FPS      = 24
LTX_FRAMES = 97      # (97-1)%8==0 ✓
LTX_STEPS  = 40
LTX_NEG    = ('worst quality, inconsistent motion, blurry, jittery, '
               'distorted, low resolution, artifacts')
LTX_PIPELINE = None


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(args: list, desc: str = '') -> bool:
    """Run ffmpeg with args list. Returns True on success."""
    try:
        r = subprocess.run(
            [FFMPEG_CMD, '-y'] + args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if r.returncode != 0:
            err = r.stderr.decode(errors='replace')[-1200:]
            print(f'FFmpeg error ({desc}):\n{err}')
        return r.returncode == 0
    except Exception as ex:
        print(f'FFmpeg launch failed ({desc}): {ex}')
        return False


def fmt_dur(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f'{h}h {m}m {sec}s' if h else f'{m}m {sec}s'


def _hex_no_hash(h: str) -> str:
    """'#aabbcc' → 'aabbcc'  (FFmpeg color format)"""
    return h.lstrip('#')


# ══════════════════════════════════════════════════════════════════════════════
# Theme from image
# ══════════════════════════════════════════════════════════════════════════════

def derive_theme(img_path: str) -> dict:
    """Derive a color palette from the image automatically."""
    try:
        img    = Image.open(img_path).resize((64, 64)).convert('RGB')
        pixels = np.array(img).reshape(-1, 3).astype(float)
        avg    = pixels.mean(axis=0)
        dark   = pixels.min(axis=0)

        # Background: very dark version of image tones
        bg_r = max(5,  min(25, int(dark[0] * 0.4)))
        bg_g = max(5,  min(25, int(dark[1] * 0.4)))
        bg_b = max(10, min(40, int(dark[2] * 0.6)))

        # Wave: boosted complement
        w_r = min(255, int(avg[2] * 1.5))
        w_g = min(255, int(avg[0] * 0.5))
        w_b = min(255, int(avg[1] * 1.9))

        # Playhead: high-contrast accent
        h_r = min(255, int(255 - avg[0]))
        h_g = min(255, int(avg[1] * 0.2))
        h_b = min(255, int(avg[2] * 0.4))

        return {
            'bg':   f'#{bg_r:02x}{bg_g:02x}{bg_b:02x}',
            'wave': f'#{w_r:02x}{w_g:02x}{w_b:02x}',
            'head': f'#{h_r:02x}{h_g:02x}{h_b:02x}',
        }
    except Exception:
        return {'bg': '#0f172a', 'wave': '#00d4ff', 'head': '#f43f5e'}


# ══════════════════════════════════════════════════════════════════════════════
# Audio duration (fast — no full decode needed for duration)
# ══════════════════════════════════════════════════════════════════════════════

def get_audio_duration(path: str) -> float:
    """Get duration in seconds using ffprobe (fast, any format)."""
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    # fallback: use FFMPEG_CMD path for ffprobe
    probe = FFMPEG_CMD.replace('ffmpeg', 'ffprobe')
    try:
        r = subprocess.run(
            [probe, '-v', 'quiet', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', path],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return float(r.stdout.strip())
    except Exception:
        pass
    # last resort: pydub
    if PYDUB_OK:
        try:
            seg = AudioSegment.from_file(path)
            return len(seg) / 1000.0
        except Exception:
            pass
    if LIBROSA_OK:
        try:
            return librosa.get_duration(path=path)
        except Exception:
            pass
    return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# LTX-Video loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_ltx():
    global LTX_PIPELINE
    if LTX_PIPELINE is not None:
        return True, ''
    if not LTX_AVAILABLE:
        return False, 'CUDA not available'
    try:
        pipe = LTXImageToVideoPipeline.from_pretrained(
            'Lightricks/LTX-Video', torch_dtype=torch.bfloat16)
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        LTX_PIPELINE = pipe
        return True, f'Loaded on {torch.cuda.get_device_name(0)}'
    except Exception as ex:
        traceback.print_exc()
        return False, str(ex)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def generate_video(audio_file, image_path, progress=None):
    """
    Full pipeline — renders entirely in FFmpeg, no Python frame loop.
    A 1h15m audio file completes in ~1-3 minutes.
    """

    def _prog(frac, desc=''):
        if progress is not None:
            progress(frac, desc=desc)
        print(f'[{int(frac*100):3d}%] {desc}')

    # ── Validate ───────────────────────────────────────────────────────────
    if audio_file is None:
        return None, '❌ Upload an audio file first'
    if image_path is None:
        return None, '❌ Upload an image first'

    audio_path = audio_file.name if hasattr(audio_file, 'name') else str(audio_file)
    img_path   = str(image_path)

    if not os.path.exists(audio_path):
        return None, f'❌ Audio not found: {audio_path}'
    if not os.path.exists(img_path):
        return None, f'❌ Image not found: {img_path}'

    # ── Duration ───────────────────────────────────────────────────────────
    _prog(0.02, 'Reading audio duration…')
    duration = get_audio_duration(audio_path)
    if duration <= 0:
        return None, '❌ Could not read audio duration'

    mb = os.path.getsize(audio_path) / 1_048_576
    _prog(0.05, f'Audio: {fmt_dur(duration)} ({mb:.1f} MB)')

    # ── Theme ──────────────────────────────────────────────────────────────
    _prog(0.06, 'Deriving color theme from image…')
    theme    = derive_theme(img_path)
    bg_hex   = _hex_no_hash(theme['bg'])
    wave_hex = _hex_no_hash(theme['wave'])

    # ── Prepare scaled background image ───────────────────────────────────
    _prog(0.08, 'Preparing background image…')
    bg_scaled = os.path.join(WORK_DIR, 'bg_scaled.png')
    _run([
        '-i', img_path,
        '-vf', (f'scale={VID_W}:{VID_H}:force_original_aspect_ratio=decrease,'
                f'pad={VID_W}:{VID_H}:(ow-iw)/2:(oh-ih)/2:color={bg_hex}'),
        bg_scaled,
    ], 'scale bg')

    # ── LTX-Video AI base (optional) ───────────────────────────────────────
    _prog(0.10, 'Checking AI video generation…')
    ai_base_video = None
    use_ltx = False

    ltx_ok, ltx_msg = _load_ltx()
    if ltx_ok:
        _prog(0.12, f'LTX-Video: generating {LTX_FRAMES} frames (~2-5 min)…')
        try:
            input_img = Image.open(img_path).convert('RGB').resize(
                (VID_W, VID_H), Image.Resampling.LANCZOS)
            result = LTX_PIPELINE(
                image=input_img,
                prompt=('Cinematic slow motion, subtle camera drift, ambient '
                        'light, dreamlike motion, high quality'),
                negative_prompt=LTX_NEG,
                width=VID_W, height=VID_H,
                num_frames=LTX_FRAMES,
                num_inference_steps=LTX_STEPS,
                guidance_scale=3.0,
            )
            _prog(0.60, 'Saving AI frames…')
            ai_silent = os.path.join(WORK_DIR, 'ltx_silent.mp4')
            export_to_video(result.frames[0], ai_silent, fps=FPS)

            # Loop AI clip to full audio duration
            _prog(0.62, 'Looping AI clip to audio duration…')
            ai_looped = os.path.join(WORK_DIR, 'ltx_looped.mp4')
            ok = _run([
                '-stream_loop', '-1',
                '-i', ai_silent,
                '-t', str(math.ceil(duration)),
                '-c:v', *_VC, '-pix_fmt', 'yuv420p',
                ai_looped,
            ], 'loop ltx')
            if ok and os.path.exists(ai_looped):
                ai_base_video = ai_looped
                use_ltx = True
                _prog(0.64, 'AI base ready')
        except Exception as ex:
            if torch.cuda.is_available():
                try: torch.cuda.empty_cache()
                except Exception: pass
            print(f'LTX failed: {ex} — using static image')
            use_ltx = False

    # ── Build background video (static image if no LTX) ───────────────────
    if not use_ltx:
        _prog(0.15, 'Building static image background video…')
        static_vid = os.path.join(WORK_DIR, 'bg_static.mp4')
        ok = _run([
            '-loop', '1',
            '-framerate', str(FPS),
            '-i', bg_scaled,
            '-t', str(math.ceil(duration)),
            '-c:v', *_VC_STILL, '-pix_fmt', 'yuv420p',
            static_vid,
        ], 'static bg video')
        if not ok:
            return None, '❌ Failed to create background video'
        ai_base_video = static_vid
        _prog(0.30, 'Background video ready')

    # ══════════════════════════════════════════════════════════════════════
    # FFmpeg waveform composite — entire waveform rendered natively.
    # showwaves filter draws the scrolling waveform directly from the audio.
    # This replaces the entire Python matplotlib frame loop.
    # Typical speed: real-time or faster (a 1h15m file finishes in ~2 min).
    # ══════════════════════════════════════════════════════════════════════
    _prog(0.65, 'Rendering waveform + compositing (FFmpeg native — fast)…')

    final = os.path.join(WORK_DIR, 'nuwave_final.mp4')

    # showwaves produces a scrolling waveform matched to the audio timeline.
    # We blend it over the background at 70% opacity using overlay+colorkey.
    wave_color_ffmpeg = wave_hex   # e.g. "00d4ff"

    # filter_complex breakdown:
    #   [0:v]               background video (AI loop or static image)
    #   [1:a]showwaves      draw waveform from audio stream
    #   scale2ref           scale waveform to match background size
    #   colorkey            make black background transparent
    #   overlay             composite waveform over background
    filter_complex = (
        f'[1:a]showwaves='
        f's={VID_W}x{VID_H//3}:'        # waveform height = bottom third
        f'mode=cline:'                    # continuous line mode (smooth)
        f'rate={FPS}:'
        f'colors={wave_color_ffmpeg}@0.9'
        f'[waves];'
        f'[waves]format=yuva420p,'
        f'colorkey=black:0.15:0.1'        # remove black bg → transparent
        f'[waves_t];'
        f'[0:v][waves_t]'
        f'overlay=0:{VID_H - VID_H//3}'  # position at bottom of frame
        f'[v]'
    )

    ok = _run([
        '-i', ai_base_video,             # [0] background video
        '-i', audio_path,                # [1] audio (for showwaves + output)
        '-filter_complex', filter_complex,
        '-map', '[v]',
        '-map', '1:a',
        '-c:v', *_VC, '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k',
        '-shortest',
        final,
    ], 'composite+encode')

    if not ok or not os.path.exists(final) or os.path.getsize(final) < 10_000:
        # Fallback: simpler showwaves without overlay (no colorkey)
        _prog(0.80, 'Retrying with simplified waveform render…')
        filter_simple = (
            f'[0:v][1:a]'
            f'[1:a]showwaves='
            f's={VID_W}x{VID_H//3}:'
            f'mode=cline:rate={FPS}:'
            f'colors={wave_color_ffmpeg}[waves];'
            f'[0:v][waves]vstack=inputs=2[v]'
        )
        # Even simpler: just render the waveform directly on a dark bg
        ok = _run([
            '-i', audio_path,
            '-filter_complex',
            (f'[0:a]showwaves='
             f's={VID_W}x{VID_H}:'
             f'mode=cline:'
             f'rate={FPS}:'
             f'colors={wave_color_ffmpeg}|{wave_color_ffmpeg}@0.4[v]'),
            '-map', '[v]',
            '-map', '0:a',
            '-c:v', *_VC, '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '192k',
            final,
        ], 'fallback showwaves')

    if not ok or not os.path.exists(final) or os.path.getsize(final) < 10_000:
        return None, '❌ FFmpeg render failed – check terminal for details'

    size_mb = os.path.getsize(final) / 1_048_576
    mode    = 'AI animation + waveform' if use_ltx else 'image background + waveform'
    _prog(1.0, 'Done!')
    return final, (
        f'✓ Video ready  |  {mode}  |  '
        f'{fmt_dur(duration)}  |  {size_mb:.1f} MB'
    )


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
import gradio as gr

DARK_CSS = """
body, .gradio-container { background:#0f172a !important; color:#94a3b8; }
footer { display:none !important; }
h1,h2,h3 { color:#00d4ff; }
"""

ltx_status = (
    '✅ **LTX-Video + CUDA detected** — AI animation background enabled.'
    if LTX_AVAILABLE else
    '⚠️ AI animation not available (CUDA missing or LTX not installed).  \n'
    'Static image background will be used — quality is still great.  \n'
    'To enable AI: `pip install torch diffusers transformers accelerate`'
)


def _sys_info():
    vm  = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.3)
    lines = [
        f'RAM: {vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB ({vm.percent}%)',
        f'CPU: {cpu}%',
    ]
    if LTX_AVAILABLE and torch.cuda.is_available():
        lines.append(f'GPU: {torch.cuda.get_device_name(0)}')
        lines.append(
            f'VRAM: {torch.cuda.memory_allocated(0)/1e9:.1f} alloc / '
            f'{torch.cuda.memory_reserved(0)/1e9:.1f} GB reserved')
    else:
        lines.append('GPU: Not available for AI video')
    return '\n'.join(lines)


def build_app():
    with gr.Blocks(title='NuWave') as demo:

        gr.Markdown('# 🎵 NuWave — AI Waveform Video Generator')
        gr.Markdown(
            'Upload audio (any size) + cover image → **Generate Video** → download MP4.  \n'
            'Theme and colors are derived automatically from your image.  \n'
            + ltx_status
        )

        with gr.Row():
            audio_in = gr.File(
                label='🎵 Audio  (MP3 / WAV / FLAC / M4A — any size)',
                file_types=['audio', '.mp3', '.wav', '.flac',
                            '.m4a', '.ogg', '.aac'],
            )
            image_in = gr.Image(
                label='🖼️ Cover Image  (JPG / PNG)',
                type='filepath',
                height=240,
            )

        gen_btn = gr.Button('🎬 Generate Video', variant='primary', size='lg')

        status = gr.Textbox(
            label='Status',
            value='Upload audio and image, then click Generate.',
            interactive=False,
            lines=2,
        )

        video_out = gr.Video(
            label='📥 Download Your Video',
            interactive=False,
        )

        with gr.Accordion('💻 System Info', open=False):
            sys_box = gr.Textbox(interactive=False, lines=4, label='')
            gr.Button('Refresh').click(fn=_sys_info, outputs=[sys_box])

        def on_generate(audio_file, image_path,
                        progress=gr.Progress(track_tqdm=True)):
            return generate_video(audio_file, image_path, progress)

        gen_btn.click(
            fn=on_generate,
            inputs=[audio_in, image_in],
            outputs=[video_out, status],
        )

        demo.load(fn=_sys_info, outputs=[sys_box])

    return demo


if __name__ == '__main__':
    # Render.com and other cloud hosts assign the port via the PORT env var.
    # Falls back to 7860 for local development.
    port = int(os.environ.get('PORT', 7860))
    app  = build_app()
    print(f'NuWave starting on port {port}')
    app.launch(
        server_name='0.0.0.0',
        server_port=port,
        share=False,
        max_file_size=None,
        allowed_paths=[WORK_DIR],
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=DARK_CSS,
    )
