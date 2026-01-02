# 🎵 Audio Visualizer - Modern Edition

A modern, beautiful audio visualizer built with PyQt6 that creates stunning waveform visualizations with optional background images.

## ✨ Features

- **Modern Upload Wizard**: Step-by-step interface for uploading audio and image files
- **Beautiful UI**: Gradient backgrounds, modern colors, and smooth animations
- **Audio Support**: Works with MP3, WAV, M4A, FLAC, OGG, AAC and more
- **Image Themes**: Apply background images that influence the waveform colors
- **Real-time Playback**: Watch the waveform scroll as your audio plays
- **Video Export**: Export your visualization as an MP4 video file
- **Seamless Workflow**: Confirm uploads before proceeding to the visualizer

## 📋 Requirements

- Python 3.8+
- PyQt6
- numpy
- librosa
- matplotlib
- Pillow
- moviepy

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

1. Run the application:
```bash
python wavefile.py
```

2. The modern upload wizard will appear:
   - **Step 1**: Click "📁 Choose Audio File" and select your audio file
   - **Step 2**: Optionally click "📁 Choose Image File" for a background theme
   - Click "Continue" to launch the visualizer

3. In the main player:
   - **▶️ Play / Pause**: Control audio playback
   - **🖼️ Change Background Image**: Update the background image anytime
   - **🎬 Export Video**: Save your visualization as an MP4 file
   - **Slider**: Seek through the audio

## 🎨 Modern Design Features

- Gradient backgrounds with smooth color transitions
- Cyan waveform that adapts to image colors
- Modern rounded buttons with hover effects
- Clean, minimalist interface
- Real-time status updates with icons
- Responsive layout

## 🌐 Deployment Ready

The application includes:
- `requirements.txt` for easy dependency management
- `.gitignore` for clean version control
- Modular code structure for easy customization

## 📝 Notes

- Audio processing supports various formats through librosa and moviepy
- Images can be PNG, JPEG, or BMP format
- Video export may take a few minutes depending on audio length
- The waveform color is influenced by the dominant color in your background image

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License - Feel free to use and modify for your projects.
