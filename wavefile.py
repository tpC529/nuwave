import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QSlider,
                             QFileDialog, QVBoxLayout, QWidget, QLabel, QHBoxLayout,
                             QFrame, QProgressBar, QStackedWidget)
from PyQt6.QtGui import QAction, QFont, QIcon
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtCore import QUrl, Qt, QTimer, QSize
from moviepy import AudioFileClip, VideoClip


class ModernUploadWizard(QWidget):
    """Modern step-by-step upload wizard for audio and image files."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Visualizer - Upload Files")
        self.setGeometry(200, 200, 600, 450)
        self.audio_path = None
        self.image_path = None
        self.samples = None
        self.sr = None
        
        # Apply modern styling
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #1a1a2e, stop:1 #16213e);
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
            QLabel#title {
                font-size: 24px;
                font-weight: bold;
                color: #00d4ff;
            }
            QLabel#subtitle {
                font-size: 16px;
                color: #aaaaaa;
            }
            QLabel#statusLabel {
                font-size: 13px;
                color: #4ade80;
                font-weight: 500;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0ea5e9, stop:1 #0284c7);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #38bdf8, stop:1 #0ea5e9);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0284c7, stop:1 #075985);
            }
            QPushButton:disabled {
                background: #374151;
                color: #6b7280;
            }
            QPushButton#secondaryBtn {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #4b5563, stop:1 #374151);
            }
            QPushButton#secondaryBtn:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #6b7280, stop:1 #4b5563);
            }
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 20px;
            }
            QProgressBar {
                border: none;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.1);
                text-align: center;
                color: white;
                height: 24px;
            }
            QProgressBar::chunk {
                border-radius: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #00d4ff, stop:1 #0ea5e9);
            }
        """)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(40, 40, 40, 40)
        
        # Title
        title = QLabel("🎵 Audio Visualizer")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("Upload your files to get started")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        layout.addSpacing(20)
        
        # Step 1: Audio Upload
        self.audio_frame = self._create_upload_frame(
            "Step 1: Upload Audio File",
            "Select an audio file (MP3, WAV, M4A, etc.)",
            "audio"
        )
        layout.addWidget(self.audio_frame)
        
        # Step 2: Image Upload
        self.image_frame = self._create_upload_frame(
            "Step 2: Upload Image (Optional)",
            "Select an image for background theme",
            "image"
        )
        layout.addWidget(self.image_frame)
        
        layout.addSpacing(10)
        
        # Progress and status
        self.overall_status = QLabel("")
        self.overall_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.overall_status)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        self.skip_image_btn = QPushButton("Skip Image")
        self.skip_image_btn.setObjectName("secondaryBtn")
        self.skip_image_btn.clicked.connect(self.skip_image)
        self.skip_image_btn.setEnabled(False)
        btn_layout.addWidget(self.skip_image_btn)
        
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.clicked.connect(self.finish_wizard)
        self.continue_btn.setEnabled(False)
        btn_layout.addWidget(self.continue_btn)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        self.setLayout(layout)
        
    def _create_upload_frame(self, title, subtitle, upload_type):
        """Create a modern upload frame for a single file type."""
        frame = QFrame()
        frame_layout = QVBoxLayout()
        frame_layout.setSpacing(12)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #00d4ff;")
        frame_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("font-size: 12px; color: #aaaaaa;")
        frame_layout.addWidget(subtitle_label)
        
        # Upload button and status
        btn_status_layout = QHBoxLayout()
        
        upload_btn = QPushButton(f"📁 Choose {upload_type.title()} File")
        upload_btn.setMinimumWidth(200)
        if upload_type == "audio":
            upload_btn.clicked.connect(self.upload_audio)
        else:
            upload_btn.clicked.connect(self.upload_image)
        btn_status_layout.addWidget(upload_btn)
        
        status_label = QLabel("Not uploaded")
        status_label.setObjectName("statusLabel")
        status_label.setStyleSheet("color: #6b7280;")
        btn_status_layout.addWidget(status_label)
        btn_status_layout.addStretch()
        
        frame_layout.addLayout(btn_status_layout)
        
        # Store references
        if upload_type == "audio":
            self.audio_status = status_label
            self.audio_btn = upload_btn
        else:
            self.image_status = status_label
            self.image_btn = upload_btn
        
        frame.setLayout(frame_layout)
        return frame
    
    def upload_audio(self):
        """Handle audio file upload."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.mp3 *.wav *.m4a *.flac *.ogg *.aac);;All Files (*.*)"
        )
        if not path:
            return
        
        self.audio_status.setText("Processing...")
        self.audio_status.setStyleSheet("color: #fbbf24;")
        QApplication.processEvents()
        
        # Process audio
        samples, sr = CenteredScrollingPlayer.process_audio(path)
        if samples is None:
            self.audio_status.setText("❌ Failed to load")
            self.audio_status.setStyleSheet("color: #ef4444;")
            self.overall_status.setText("Failed to load audio file. Please try another file.")
            self.overall_status.setStyleSheet("color: #ef4444;")
            return
        
        self.audio_path = path
        self.samples = samples
        self.sr = sr
        
        # Update UI
        filename = path.split('/')[-1]
        self.audio_status.setText(f"✓ {filename}")
        self.audio_status.setStyleSheet("color: #4ade80; font-weight: bold;")
        self.overall_status.setText("Audio file loaded successfully! Now upload an image or continue.")
        self.overall_status.setStyleSheet("color: #4ade80;")
        
        # Enable next steps
        self.image_btn.setEnabled(True)
        self.skip_image_btn.setEnabled(True)
        self.continue_btn.setEnabled(True)
        
    def upload_image(self):
        """Handle image file upload."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )
        if not path:
            return
        
        self.image_status.setText("Loading...")
        self.image_status.setStyleSheet("color: #fbbf24;")
        QApplication.processEvents()
        
        # Validate image
        try:
            img = Image.open(path)
            img.verify()
            self.image_path = path
            
            filename = path.split('/')[-1]
            self.image_status.setText(f"✓ {filename}")
            self.image_status.setStyleSheet("color: #4ade80; font-weight: bold;")
            self.overall_status.setText("All files uploaded! Click Continue to start visualizing.")
            self.overall_status.setStyleSheet("color: #4ade80;")
        except Exception as e:
            self.image_status.setText("❌ Invalid image")
            self.image_status.setStyleSheet("color: #ef4444;")
            self.overall_status.setText(f"Failed to load image: {str(e)}")
            self.overall_status.setStyleSheet("color: #ef4444;")
    
    def skip_image(self):
        """Skip image upload and continue."""
        self.image_status.setText("⊘ Skipped")
        self.image_status.setStyleSheet("color: #6b7280;")
        self.overall_status.setText("Image skipped. Click Continue to start visualizing.")
        self.overall_status.setStyleSheet("color: #4ade80;")
    
    def finish_wizard(self):
        """Complete the wizard and proceed."""
        if self.audio_path is None:
            self.overall_status.setText("Please upload an audio file first!")
            self.overall_status.setStyleSheet("color: #ef4444;")
            return
        self.accept()
    
    def accept(self):
        """Mark wizard as completed and close."""
        self.done(1)


class CenteredScrollingPlayer(QMainWindow):
    def __init__(self, samples=None, sr=None):
        super().__init__()
        self.setWindowTitle("🎵 Audio Visualizer - Modern Player")
        self.setGeometry(100, 100, 1400, 800)

        self.window_sec = 10.0  # Total visible window (±5 seconds)
        self.half_window = self.window_sec / 2

        self.player = QMediaPlayer()
        # audio output is required in Qt6 to actually hear audio
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        # sensible default volume (0.0 - 1.0)
        try:
            self.audio_output.setVolume(0.8)
        except Exception:
            pass

        # Apply modern styling to main window
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0f172a, stop:1 #1e293b);
            }
            QWidget {
                background: transparent;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0ea5e9, stop:1 #0284c7);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #38bdf8, stop:1 #0ea5e9);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #0284c7, stop:1 #075985);
            }
            QSlider::groove:horizontal {
                border: none;
                height: 8px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                           stop:0 #00d4ff, stop:1 #0ea5e9);
                border: none;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #00d4ff;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                           stop:0 #00d4ff, stop:1 #0ea5e9);
                border-radius: 4px;
            }
            QLabel {
                color: #e2e8f0;
                font-size: 13px;
            }
            QMenuBar {
                background: rgba(15, 23, 42, 0.8);
                color: #ffffff;
                padding: 4px;
            }
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: rgba(14, 165, 233, 0.3);
            }
            QMenu {
                background: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #0ea5e9;
            }
        """)

        # UI
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        self.canvas = FigureCanvas(plt.Figure(figsize=(14, 8)))
        layout.addWidget(self.canvas, stretch=1)
        # update background placement when canvas resizes
        self.canvas.mpl_connect('resize_event', lambda evt: self._place_background() if getattr(self, 'bg_pil', None) is not None else None)

        # Modern control panel
        controls = QWidget()
        ctrl_layout = QVBoxLayout(controls)
        ctrl_layout.setSpacing(12)
        
        # Playback controls row
        playback_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶️ Play / Pause")
        self.play_btn.setMinimumWidth(150)
        playback_layout.addWidget(self.play_btn)
        playback_layout.addStretch()
        
        self.load_image_btn = QPushButton("🖼️ Change Background Image")
        playback_layout.addWidget(self.load_image_btn)
        
        self.export_btn = QPushButton("🎬 Export Video")
        playback_layout.addWidget(self.export_btn)
        
        ctrl_layout.addLayout(playback_layout)
        
        # Progress slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimumHeight(30)
        ctrl_layout.addWidget(self.slider)
        
        # Status label
        self.status = QLabel("Ready to play")
        self.status.setStyleSheet("color: #4ade80; font-weight: 500; font-size: 14px;")
        ctrl_layout.addWidget(self.status)
        
        layout.addWidget(controls)

        self.setCentralWidget(central)

        # Plot setup with modern dark theme
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_facecolor('#0f172a')
        self.canvas.figure.patch.set_facecolor('#0f172a')
        self.line, = self.ax.plot([], [], color='#00d4ff', lw=2)
        self.playhead = self.ax.axvline(0, color='#f43f5e', lw=2.5, ls='--', alpha=0.9)
        # show from 0..window_sec initially (cursor at start)
        self.ax.set_xlim(0, self.window_sec)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlabel("Time (s)", color='#94a3b8', fontsize=11)
        self.ax.tick_params(colors='#94a3b8', labelsize=10)
        self.ax.grid(alpha=0.15, color='#334155')
        self.canvas.figure.tight_layout()

        # Data
        self.samples = samples
        self.sr = sr
        self.bg_image = None

        # Connections
        self.play_btn.clicked.connect(self.toggle_play)
        self.export_btn.clicked.connect(self.export_video)
        self.load_image_btn.clicked.connect(self.load_image)
        self.slider.sliderMoved.connect(lambda p: self.player.setPosition(p))
        self.player.positionChanged.connect(self.update_plot)
        self.player.durationChanged.connect(lambda d: self.slider.setMaximum(d))

        # Menu & shortcuts
        load_act = QAction("Load Audio", self)
        load_act.setShortcut("Ctrl+O")
        load_act.triggered.connect(self.load_file)
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        file_menu.addAction(load_act)
        self.play_btn.setShortcut("Space")

        # If samples were provided at construction, initialize slider and plot
        if self.samples is not None and self.sr is not None:
            duration_ms = int(len(self.samples) / self.sr * 1000)
            self.slider.setMinimum(0)
            self.slider.setMaximum(duration_ms)
            self.slider.setValue(0)
            self.player.setPosition(0)
            self.update_plot(0)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.*)")
        if not path:
            return

        self.status.setText("⏳ Loading audio...")
        self.status.setStyleSheet("color: #fbbf24; font-weight: 500;")
        QApplication.processEvents()
        
        samples, sr = self.process_audio(path)
        if samples is None:
            self.status.setText(f"❌ Failed to load: {path}")
            self.status.setStyleSheet("color: #ef4444; font-weight: 500;")
            return

        self.samples = samples
        self.sr = float(sr)

        self.player.setSource(QUrl.fromLocalFile(path))
        # set slider range based on duration and show initial waveform
        duration_ms = int(len(self.samples) / self.sr * 1000)
        self.slider.setMinimum(0)
        self.slider.setMaximum(duration_ms)
        self.slider.setValue(0)
        self.player.setPosition(0)
        filename = path.split('/')[-1]
        self.status.setText(f"✓ Loaded: {filename}")
        self.status.setStyleSheet("color: #4ade80; font-weight: 500;")
        self.update_plot(0)

    def prompt_load(self):
        # Called after the window shows to ask the user for a file
        self.load_file()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if not path:
            return
        try:
            self.apply_theme_from_image(path)
            filename = path.split('/')[-1]
            self.status.setText(f"✓ Image applied: {filename}")
            self.status.setStyleSheet("color: #4ade80; font-weight: 500;")
        except Exception as e:
            self.status.setText(f"❌ Failed to apply image: {e}")
            self.status.setStyleSheet("color: #ef4444; font-weight: 500;")

    def apply_theme_from_image(self, img_path):
        # compute dominant color
        img = Image.open(img_path).convert('RGBA')
        small = img.resize((64, 64))
        result = small.convert('RGB').getcolors(64*64)
        if not result:
            return
        result.sort(key=lambda x: x[0], reverse=True)
        dominant = result[0][1]
        dom_rgb = tuple([c/255.0 for c in dominant])
        # compute audio brightness factor from RMS
        rms = 0.0
        if self.samples is not None:
            rms = float(np.sqrt(np.mean(self.samples.astype(np.float64)**2)))
        bright = 0.6 + min(1.0, rms*5.0)

        # set waveform color influenced by dominant color and audio brightness
        wave_rgb = tuple(min(1.0, c * bright) for c in dom_rgb)
        bg_lum = 0.2126*dom_rgb[0] + 0.7152*dom_rgb[1] + 0.0722*dom_rgb[2]
        
        # Modern dark theme approach
        if bg_lum < 0.5:
            # Keep dark background with modern colors
            bg_color = '#0f172a'
            text_color = '#94a3b8'
        else:
            # Darker background for light dominant colors
            bg_color = '#1e293b'
            text_color = '#cbd5e1'

        self.ax.set_facecolor(bg_color)
        self.canvas.figure.set_facecolor(bg_color)
        self.ax.tick_params(colors=text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.line.set_color(wave_rgb)
        self.playhead.set_color('#f43f5e')
        self.canvas.draw_idle()
        # also apply image as background (keeps aspect ratio)
        try:
            self.apply_background_image(img_path)
        except Exception:
            pass

    def apply_background_image(self, img_path):
        img = Image.open(img_path).convert('RGBA')
        arr = np.asarray(img)
        # store PIL image and array for later placement/resizing
        self.bg_pil = img
        self.bg_arr = arr
        # remove existing bg image if present
        if getattr(self, 'bg_image', None) is not None:
            try:
                self.bg_image.remove()
            except Exception:
                pass
            self.bg_image = None
        # place background according to current axes limits
        self._place_background()

    @staticmethod
    def process_audio(path):
        """Try to load audio with librosa, fallback to moviepy if needed.

        Returns (samples: np.ndarray, sr: int) or (None, None) on failure.
        """
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception:
            # Fallback to moviepy (uses ffmpeg) for formats like m4a
            try:
                clip = AudioFileClip(path)
                sr = int(clip.fps)
                arr = clip.to_soundarray()
                clip.close()
                # convert to mono if stereo
                if arr.ndim == 2:
                    arr = arr.mean(axis=1)
                # moviepy returns float in [-1,1], ensure float32
                return np.asarray(arr, dtype=np.float32), sr
            except Exception:
                return None, None

    def _place_background(self):
        """Place or update the background image to preserve aspect ratio inside axes."""
        if getattr(self, 'bg_pil', None) is None:
            return

        # remove old image artist
        if getattr(self, 'bg_image', None) is not None:
            try:
                self.bg_image.remove()
            except Exception:
                pass
            self.bg_image = None

        # current axes data limits
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        axis_w = x1 - x0
        axis_h = y1 - y0 if (y1 - y0) != 0 else 1.0

        iw, ih = self.bg_pil.size
        img_ratio = iw / ih if ih != 0 else 1.0
        axis_ratio = axis_w / axis_h

        # fit image to axes while preserving aspect ratio
        if img_ratio >= axis_ratio:
            # image is wider (relative); fit width to axis width
            target_w = axis_w
            target_h = axis_w / img_ratio
        else:
            # image is taller; fit height to axis height
            target_h = axis_h
            target_w = axis_h * img_ratio

        x0_img = x0 + (axis_w - target_w) / 2.0
        x1_img = x0_img + target_w
        y0_img = y0 + (axis_h - target_h) / 2.0
        y1_img = y0_img + target_h

        # create image artist behind waveform
        self.bg_image = self.ax.imshow(self.bg_arr, extent=(x0_img, x1_img, y0_img, y1_img), aspect='auto', zorder=0, alpha=0.6)
        self.line.set_zorder(2)
        self.playhead.set_zorder(3)
        self.canvas.draw_idle()

    def toggle_play(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def update_plot(self, pos_ms):
        if self.samples is None: return
        current_sec = pos_ms / 1000.0

        # sliding window anchored to start (show 0..window or current-centered window when past half)
        start_sec = max(0.0, current_sec - self.half_window)
        end_sec = start_sec + self.window_sec

        start_idx = max(0, int(start_sec * self.sr))
        end_idx = min(len(self.samples), int(end_sec * self.sr))

        visible = self.samples[start_idx:end_idx]
        if len(visible) < 2: return

        # t = absolute time positions for visible samples
        t = np.linspace(start_idx / self.sr, (start_idx + len(visible)) / self.sr, len(visible))
        self.line.set_data(t, visible)

        # update axis limits so the left edge is 0 initially
        self.ax.set_xlim(start_sec, end_sec)

        # update playhead to current absolute time
        try:
            self.playhead.set_xdata((current_sec, current_sec))
        except Exception:
            self.playhead = self.ax.axvline(current_sec, color='red', lw=2, ls='--')

        # reposition background to respect new axis limits (if present)
        if getattr(self, 'bg_pil', None) is not None:
            self._place_background()

        # Dynamic vertical fill
        margin = 0.1
        amp_max = np.max(np.abs(visible))
        if amp_max > 0:
            self.ax.set_ylim(-amp_max * (1 + margin), amp_max * (1 + margin))

        self.slider.blockSignals(True)
        self.slider.setValue(pos_ms)
        self.slider.blockSignals(False)

        self.canvas.draw_idle()

    def make_frame(self, t):
        pos_ms = int(t * 1000)
        self.update_plot(pos_ms)
        img = np.frombuffer(self.canvas.tostring_rgb(), dtype='uint8')
        img = img.reshape(self.canvas.figure.canvas.get_width_height()[::-1] + (3,))
        return img

    def export_video(self):
        if self.samples is None: return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Video", "", "MP4 (*.mp4)")
        if not out_path: return

        self.status.setText("⏳ Exporting video... This may take a few minutes")
        self.status.setStyleSheet("color: #fbbf24; font-weight: 500;")
        QApplication.processEvents()
        
        audio = AudioFileClip(self.player.source().toLocalFile())
        video = VideoClip(self.make_frame, duration=self.player.duration() / 1000)
        video = video.set_audio(audio)
        video.write_videofile(out_path, fps=30, codec='libx264', audio_codec='aac')
        
        filename = out_path.split('/')[-1]
        self.status.setText(f"✓ Exported: {filename}")
        self.status.setStyleSheet("color: #4ade80; font-weight: 500;")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # Show modern upload wizard
    wizard = ModernUploadWizard()
    result = wizard.exec()
    
    if result != 1 or wizard.audio_path is None:
        sys.exit(0)
    
    # Create main player with uploaded files
    win = CenteredScrollingPlayer(samples=wizard.samples, sr=wizard.sr)
    win.player.setSource(QUrl.fromLocalFile(wizard.audio_path))
    
    # Apply image theme if uploaded
    if wizard.image_path:
        try:
            win.apply_theme_from_image(wizard.image_path)
            filename = wizard.image_path.split('/')[-1]
            win.status.setText(f"✓ Ready to play - Image: {filename}")
            win.status.setStyleSheet("color: #4ade80; font-weight: 500;")
        except Exception as e:
            print(f"Failed to apply background image: {e}")
    else:
        win.status.setText("✓ Ready to play - No background image")
        win.status.setStyleSheet("color: #4ade80; font-weight: 500;")
    
    win.show()
    sys.exit(app.exec())