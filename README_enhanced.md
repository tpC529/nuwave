# 🎵 Audio Visualizer with Video Generation - Modern Web Edition

A cutting-edge, web-based audio visualizer built with NiceGUI that generates stylized videos using Stable Video Diffusion (SVD), overlays dynamic waveforms via FFmpeg, and provides a sleek GUI for scrubbing through episodes and downloading locally.

## ✨ Features

- **AI Video Generation**: Uses Stable Video Diffusion to create animated videos from uploaded images, synced with audio.
- **Dynamic Waveforms**: FFmpeg-powered waveforms overlaid on generated videos for immersive visuals.
- **Modern GUI**: Sleek interface with file uploads, real-time scrubbing, playback controls, and download options.
- **Web Upload Interface**: Drag-and-drop audio and image files.
- **Audio Support**: MP3, WAV, M4A, FLAC, OGG, AAC.
- **Image Themes**: Background images influence waveform colors and video generation.
- **Real-time Playback**: Scrub through the episode with a slider.
- **Video Export & Download**: Generate and download MP4 videos locally.
- **Customization**: Waveform styles, colors, themes, and export titles.

## 📋 Requirements

- Python 3.8+
- nicegui, numpy, librosa, matplotlib, Pillow, moviepy, psutil, mutagen, imageio, torch, diffusers, transformers, accelerate
- FFmpeg installed
- GPU recommended for SVD (uses CUDA if available)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/tpC529/audiovisualizer.git
cd audiovisualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Run the web server:
```bash
python wavefile_enhanced.py
```

2. Open http://localhost:8080

3. Upload files:
   - **Audio**: MP3/WAV for waveform and sync.
   - **Image**: For video generation and theme.

4. Generate Video:
   - Click "🎬 Generate Video with SVD" to create animated video with waveforms.

5. Visualize & Control:
   - Play/pause, stop, scrub with slider.
   - Customize waveforms (style, color, thickness).
   - Export & download MP4.

## 🎨 Advanced Features

- **SVD Integration**: Generates 25-frame videos from images using Hugging Face diffusers.
- **FFmpeg Waveforms**: Real-time waveform overlays with filters.
- **Scrub Functionality**: Slider updates plot and audio position.
- **Download Ready**: Videos saved locally as MP4.

## 🌐 Deployment

Run as web server; deploy to any Python-supporting platform.

## 🤝 Contributing

PRs welcome for improvements!

## 📄 License

MIT License.