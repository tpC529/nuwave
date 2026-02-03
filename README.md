# 🎵 Audio Visualizer - Web Edition

A modern, web-based audio visualizer built with NiceGUI that creates stunning waveform visualizations with optional background images.

## ✨ Features

- **Web Upload Interface**: Drag-and-drop or click to upload audio and image files
- **Beautiful UI**: Gradient backgrounds, modern colors, and smooth animations
- **Audio Support**: Works with MP3, WAV, M4A, FLAC, OGG, AAC and more
- **Image Themes**: Apply background images that influence the waveform colors
- **Real-time Playback**: Watch the waveform scroll as your audio plays
- **Video Export**: Export your visualization as an MP4 video file
- **Seamless Workflow**: Upload files and start visualizing immediately

## 📋 Requirements

- Python 3.8+
- nicegui
- numpy
- librosa
- matplotlib
- Pillow
- moviepy

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/audiovisualizer.git
cd audiovisualizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. Run the web server:
```bash
python wavefile.py
```

2. Open your browser to http://localhost:8080

3. Upload your files:
   - **Upload Audio**: Select an audio file (MP3, WAV, M4A, etc.)
   - **Upload Image**: Optionally select an image for background theme

4. Control playback:
   - **▶️ Play / Pause**: Control audio playback
   - **🎬 Export Video**: Save your visualization as an MP4 file
   - **Slider**: Seek through the audio

## 🎨 Modern Design Features

- Gradient backgrounds with smooth color transitions
- Cyan waveform that adapts to image colors
- Modern rounded buttons with hover effects
- Clean, minimalist interface
- Real-time status updates
- Responsive web layout

## 🌐 Web Deployment Ready

The application runs as a web server and can be deployed to any platform supporting Python.

## 📝 Notes

- Audio processing supports various formats through librosa and moviepy
- Images can be PNG, JPEG, or BMP format
- Video export may take a few minutes depending on audio length
- The waveform color is influenced by the dominant color in your background image

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

## 📄 License

MIT License - Feel free to use and modify for your projects.
