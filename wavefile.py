import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from PIL import Image
from nicegui import ui
from nicegui import events
import ffmpeg
import psutil
from pydub import AudioSegment
from mutagen import File as MutagenFile
import imageio
import imageio_ffmpeg
import json
import asyncio
import threading
import tempfile
import base64

# Configure bundled FFmpeg from imageio_ffmpeg globally before any use
try:
    _ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    os.environ['FFMPEG_BINARY'] = _ffmpeg_path
    os.environ['PATH'] = os.path.dirname(_ffmpeg_path) + os.pathsep + os.environ.get('PATH', '')
    AudioSegment.converter = _ffmpeg_path
    AudioSegment.ffmpeg = _ffmpeg_path
except Exception as e:
    print(f"Warning: Could not set up bundled FFmpeg: {e}")

CONFIG_FILE = 'audiovisualizer_config.json'
CURRENT_VISUALIZER = None

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except:
        pass

class AudioVisualizer:
    def __init__(self):
        self.samples = None
        self.sr = None
        self.audio_path = None
        self.image_path = None
        self.current_time = 0
        self.duration = 0
        self.is_playing = False
        self.window_sec = 10.0
        self.half_window = self.window_sec / 2

        self.audio_b64 = None
        self.audio_type = None
        self.visualizer_opened = False
        self.last_export_path = None
        self.see_episode_button = None

        # Use the globally configured FFmpeg path
        try:
            self.ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            self.ffmpeg_cmd = 'ffmpeg'

        # Upload readiness
        self.audio_ready = False
        self.image_ready = False

        # Customization
        self.wave_style = 'Line'
        self.wave_color = '#00d4ff'
        self.wave_thickness = 2
        self.wave_opacity = 1.0
        self.animation_speed = 1.0
        self.current_theme = 'Default'

        self.themes = {
            'Default': {'bg': '#0f172a', 'text': '#94a3b8', 'wave': '#00d4ff', 'playhead': '#f43f5e'},
            'Dark Blue': {'bg': '#1e293b', 'text': '#cbd5e1', 'wave': '#3b82f6', 'playhead': '#ef4444'},
            'Green': {'bg': '#0f172a', 'text': '#94a3b8', 'wave': '#10b981', 'playhead': '#f59e0b'},
            'Purple': {'bg': '#2d1b69', 'text': '#a78bfa', 'wave': '#8b5cf6', 'playhead': '#f97316'},
        }

        self.config = load_config()
        self.load_config()

        self.playback_timer = None

    def load_config(self):
        self.config = load_config()
        self.wave_style = self.config.get('wave_style', 'Line')
        self.wave_color = self.config.get('wave_color', '#00d4ff')
        self.wave_thickness = self.config.get('wave_thickness', 2)
        self.wave_opacity = self.config.get('wave_opacity', 1.0)
        self.animation_speed = self.config.get('animation_speed', 1.0)
        self.current_theme = self.config.get('current_theme', 'Default')

    def save_config(self):
        self.config['wave_style'] = self.wave_style
        self.config['wave_color'] = self.wave_color
        self.config['wave_thickness'] = self.wave_thickness
        self.config['wave_opacity'] = self.wave_opacity
        self.config['animation_speed'] = self.animation_speed
        self.config['current_theme'] = self.current_theme
        save_config(self.config)

    @staticmethod
    def process_audio(path):
        """Try to load audio with librosa, fallback to moviepy if needed."""
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception:
            try:
                audio = AudioSegment.from_file(path)
                sr = int(audio.frame_rate)
                samples = np.array(audio.get_array_of_samples())
                # handle stereo
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
                # normalize according to sample width
                max_val = float(2 ** (8 * audio.sample_width - 1))
                samples = samples.astype(np.float32) / max_val
                return samples, sr
            except Exception:
                return None, None

    def create_plot(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = self.themes[self.current_theme]
        ax.set_facecolor(colors['bg'])
        fig.patch.set_facecolor(colors['bg'])
        ax.tick_params(colors=colors['text'])
        ax.xaxis.label.set_color(colors['text'])
        ax.grid(alpha=0.15, color='#334155')

        # Add image background if available
        if self.image_path and os.path.exists(self.image_path):
            try:
                img = Image.open(self.image_path).convert('RGBA')
                img_array = np.array(img)
                # Resize to fit the plot area
                img_resized = img.resize((1200, 600), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
                # Display as background with low alpha
                ax.imshow(img_array, extent=[0, self.window_sec, -1.1, 1.1], aspect='auto', alpha=0.2, zorder=-1)
            except Exception:
                pass

        line, = ax.plot([], [], color=colors['wave'], lw=self.wave_thickness, alpha=self.wave_opacity)
        playhead = ax.axvline(0, color=colors['playhead'], lw=2.5, ls='--', alpha=0.9)

        ax.set_xlim(0, self.window_sec)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel("Time (s)", color=colors['text'], fontsize=11)

        return fig, ax, line, playhead

    def build_visualizer_ui(self):
        ui.label('🎵 Audio Visualizer').style('font-size: 24px; font-weight: bold; color: #00d4ff; text-align: center;')

        # Plot
        fig, ax, line, playhead = self.create_plot()
        self.ax = ax
        self.line = line
        self.playhead = playhead
        self.plot = ui.pyplot()
        self.plot.figure = fig

        # Audio element
        self.audio_element = ui.html('<audio id="audio" controls style="width:100%;"></audio>', sanitize=False)
        if self.audio_b64 and self.audio_type:
            self.audio_element.content = f'''
            <audio id="audio" controls style="width:100%;" ontimeupdate="updateTime(this.currentTime)">
            <source src="data:{self.audio_type};base64,{self.audio_b64}" type="{self.audio_type}">
            </audio>
            <script>
            function updateTime(time) {{
                // This will be handled by NiceGUI events
            }}
            </script>
            '''

        # Controls
        with ui.row():
            ui.button('▶️ Play / Pause', on_click=self.toggle_play)
            ui.button('⏹️ Stop', on_click=self.stop_audio)
            ui.button('💾 Export & Download MP4', on_click=self.export_and_download)

        # Slider and Volume
        with ui.row():
            self.slider = ui.slider(min=0, max=100, value=0, on_change=lambda e: self.on_slider_change(e.value))
            self.volume_slider = ui.slider(min=0, max=100, value=100, on_change=lambda e: self.set_volume(e.value))

        # Status
        self.status_label = ui.label("Visualizer loading...").style('color: #4ade80; font-weight: 500;')

        # Customization
        with ui.expansion('Waveform Customization').classes('w-full'):
            with ui.row():
                ui.select(['Line', 'Bars', 'Dots', 'Filled'], value='Line', on_change=lambda e: self.update_style(e.value)).props('label=Style')
                ui.color_input(value=self.wave_color, on_change=lambda e: self.choose_color(e.value)).props('label=Color')

            with ui.row():
                ui.slider(min=1, max=10, value=self.wave_thickness, on_change=lambda e: self.update_thickness(e.value)).props('label=Thickness')
                ui.slider(min=1, max=100, value=int(self.wave_opacity * 100), on_change=lambda e: self.update_opacity(e.value)).props('label=Opacity')

            with ui.row():
                ui.slider(min=50, max=200, value=int(self.animation_speed * 100), on_change=lambda e: self.update_speed(e.value)).props('label=Animation Speed')
                ui.select(list(self.themes.keys()), value=self.current_theme, on_change=lambda e: self.apply_theme(e.value)).props('label=Theme')

    def update_plot(self, current_sec):
        if self.samples is None or self.plot is None:
            return

        start_sec = max(0.0, current_sec - self.half_window)
        end_sec = start_sec + self.window_sec

        start_idx = max(0, int(start_sec * self.sr))
        end_idx = min(len(self.samples), int(end_sec * self.sr))

        visible = self.samples[start_idx:end_idx]
        if len(visible) < 2:
            return

        t = np.linspace(start_idx / self.sr, (start_idx + len(visible)) / self.sr, len(visible))
        self.line.set_data(t, visible)
        self.ax.set_xlim(start_sec, end_sec)
        self.playhead.set_xdata((current_sec, current_sec))

        margin = 0.1
        amp_max = np.max(np.abs(visible))
        if amp_max > 0:
            self.ax.set_ylim(-amp_max * (1 + margin), amp_max * (1 + margin))

        self.plot.update()

    def prepare_visualization(self, open_window: bool = True):
        """Ensure UI and plot are configured once audio and image are loaded."""
        if not (self.audio_ready and self.image_ready):
            return
        # Update duration/slider if samples present
        if self.samples is not None:
            self.duration = len(self.samples) / self.sr if self.sr else 0
            if hasattr(self, 'slider') and self.slider is not None:
                self.slider._props['max'] = self.duration
                self.slider.value = 0
                self.slider.update()

        # Apply image-based theme if image is present
        if getattr(self, 'image_path', None):
            try:
                self.apply_theme_from_image(self.image_path)
            except Exception:
                pass

        # Redraw plot at start
        try:
            self.update_plot(0)
        except Exception:
            pass

        # Reveal visualizer and hide loading indicator
        try:
            if hasattr(self, 'visualizer_container') and self.visualizer_container is not None:
                self.visualizer_container.classes(remove='hidden')
            if hasattr(self, 'loading_indicator') and self.loading_indicator is not None:
                self.loading_indicator.classes(add='hidden')
        except Exception:
            pass

        if hasattr(self, 'status_label') and self.status_label is not None:
            self.status_label.text = "✓ Visualizer ready"
        if self.see_episode_button is not None:
            self.see_episode_button.enable()

        global CURRENT_VISUALIZER
        CURRENT_VISUALIZER = self
        if open_window and not self.visualizer_opened:
            self.visualizer_opened = True
            ui.run_javascript("window.open('/visualizer','_blank')")

    def apply_theme_from_image(self, img_path):
        img = Image.open(img_path).convert('RGBA')
        small = img.resize((64, 64))
        result = small.convert('RGB').getcolors(64*64)
        if not result:
            return
        result.sort(key=lambda x: x[0], reverse=True)
        dominant = result[0][1]
        dom_rgb = tuple([c/255.0 for c in dominant])

        rms = 0.0
        if self.samples is not None:
            rms = float(np.sqrt(np.mean(self.samples.astype(np.float64)**2)))
        bright = 0.6 + min(1.0, rms*5.0)

        wave_rgb = tuple(min(1.0, c * bright) for c in dom_rgb)
        bg_lum = 0.2126*dom_rgb[0] + 0.7152*dom_rgb[1] + 0.0722*dom_rgb[2]

        if bg_lum < 0.5:
            bg_color = '#0f172a'
            text_color = '#94a3b8'
        else:
            bg_color = '#1e293b'
            text_color = '#cbd5e1'

        self.image_path = img_path

        if not hasattr(self, 'ax') or self.ax is None or self.plot is None:
            return

        self.ax.set_facecolor(bg_color)
        self.plot.fig.patch.set_facecolor(bg_color)
        self.ax.tick_params(colors=text_color)
        self.ax.xaxis.label.set_color(text_color)
        self.wave_color = '#%02x%02x%02x' % (int(wave_rgb[0]*255), int(wave_rgb[1]*255), int(wave_rgb[2]*255))
        self.line.set_color(wave_rgb)
        self.playhead.set_color('#f43f5e')
        # Re-add image background
        try:
            img = Image.open(self.image_path).convert('RGBA')
            img_array = np.array(img)
            img_resized = img.resize((1200, 600), Image.Resampling.LANCZOS)
            img_array = np.array(img_resized)
            # Clear previous images
            for img_artist in self.ax.images:
                img_artist.remove()
            self.ax.imshow(img_array, extent=[0, self.window_sec, -1.1, 1.1], aspect='auto', alpha=0.2, zorder=-1)
        except Exception:
            pass
        self.plot.update()

    async def on_audio_upload(self, e: events.UploadEventArguments):
        if e.content is None:
            return

        # Save uploaded file temporarily
        if hasattr(self, 'loading_indicator') and self.loading_indicator is not None:
            self.loading_indicator.classes(remove='hidden')
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(e.name)[1]) as tmp:
            tmp.write(e.content.read())
            self.audio_path = tmp.name

        self.status_label.text = "⏳ Processing audio..."
        await asyncio.sleep(0.1)

        samples, sr = self.process_audio(self.audio_path)
        if samples is None:
            self.status_label.text = "❌ Failed to load audio (install FFmpeg or imageio-ffmpeg)"
            return

        self.samples = samples
        self.sr = sr
        self.duration = len(samples) / sr
        self.current_time = 0
        self.audio_ready = True

        # Update slider
        self.slider._props['max'] = self.duration
        self.slider.update()

        # Create audio element with base64 data
        with open(self.audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode()
        audio_type = 'audio/wav' if self.audio_path.endswith('.wav') else 'audio/mpeg'
        self.audio_b64 = audio_data
        self.audio_type = audio_type

        if hasattr(self, 'audio_element') and self.audio_element is not None:
            self.audio_element.content = f'''
            <audio id="audio" controls style="width:100%;" ontimeupdate="updateTime(this.currentTime)">
            <source src="data:{self.audio_type};base64,{self.audio_b64}" type="{self.audio_type}">
            </audio>
            <script>
            function updateTime(time) {{
                // This will be handled by NiceGUI events
            }}
            </script>
            '''

        filename = e.name
        self.status_label.text = f"✓ Loaded audio: {filename}"
        # Configure visualization now that audio is loaded
        try:
            self.prepare_visualization(open_window=True)
        except Exception:
            pass

    async def on_image_upload(self, e: events.UploadEventArguments):
        if e.content is None:
            return

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            tmp.write(e.content.read())
            img_path = tmp.name

        try:
            if hasattr(self, 'loading_indicator') and self.loading_indicator is not None:
                self.loading_indicator.classes(remove='hidden')
            self.apply_theme_from_image(img_path)
            filename = e.name
            self.status_label.text = f"✓ Image applied: {filename}"
            self.image_ready = True
            # If audio already loaded, configure the visualization
            try:
                self.prepare_visualization(open_window=True)
            except Exception:
                pass
        except Exception as ex:
            self.status_label.text = f"❌ Failed to apply image: {str(ex)}"

    def toggle_play(self):
        if self.is_playing:
            ui.run_javascript('document.getElementById("audio").pause();')
            self.stop_playback_timer()
            self.is_playing = False
        else:
            ui.run_javascript('document.getElementById("audio").play();')
            self.start_playback_timer()
            self.is_playing = True

    def stop_audio(self):
        ui.run_javascript('document.getElementById("audio").pause(); document.getElementById("audio").currentTime = 0;')
        self.stop_playback_timer()
        self.is_playing = False
        self.current_time = 0
        self.slider.value = 0
        self.update_plot(0)

    def update_time(self, time):
        self.current_time = time
        self.slider.value = time
        self.update_plot(time)

    def on_slider_change(self, value):
        self.current_time = value
        # Update audio position via JavaScript
        ui.run_javascript(f'document.getElementById("audio").currentTime = {value};')
        self.update_plot(value)

    def set_volume(self, value):
        volume = value / 100.0
        ui.run_javascript(f'document.getElementById("audio").volume = {volume};')

    def update_style(self, style):
        self.wave_style = style
        # For simplicity, keep as line for now
        # Could implement different plot types
        self.update_plot(self.current_time)

    def choose_color(self, color):
        self.wave_color = color
        self.line.set_color(color)
        self.plot.update()

    def update_thickness(self, val):
        self.wave_thickness = val
        self.line.set_linewidth(val)
        self.plot.update()

    def update_opacity(self, val):
        self.wave_opacity = val / 100.0
        self.line.set_alpha(self.wave_opacity)
        self.plot.update()

    def update_speed(self, val):
        self.animation_speed = val / 100.0
        self.window_sec = 10.0 * self.animation_speed
        self.half_window = self.window_sec / 2
        self.update_plot(self.current_time)

    def apply_theme(self, theme):
        self.current_theme = theme
        colors = self.themes[theme]
        self.ax.set_facecolor(colors['bg'])
        self.plot.fig.patch.set_facecolor(colors['bg'])
        self.ax.tick_params(colors=colors['text'])
        self.ax.xaxis.label.set_color(colors['text'])
        self.line.set_color(colors['wave'])
        self.playhead.set_color(colors['playhead'])
        # Re-add image if present
        if self.image_path:
            try:
                img = Image.open(self.image_path).convert('RGBA')
                img_array = np.array(img)
                img_resized = img.resize((1200, 600), Image.Resampling.LANCZOS)
                img_array = np.array(img_resized)
                for img_artist in self.ax.images:
                    img_artist.remove()
                self.ax.imshow(img_array, extent=[0, self.window_sec, -1.1, 1.1], aspect='auto', alpha=0.2, zorder=-1)
            except Exception:
                pass
        self.plot.update()

    async def update_playback(self):
        current_time = await ui.run_javascript('document.getElementById("audio")?.currentTime || 0')
        self.current_time = current_time
        self.slider.value = current_time
        self.update_plot(current_time)

    def start_playback_timer(self):
        if self.playback_timer:
            self.playback_timer.cancel()
        self.playback_timer = ui.timer(0.1, self.update_playback, once=False)

    def stop_playback_timer(self):
        if self.playback_timer:
            self.playback_timer.cancel()
            self.playback_timer = None

    def update_memory(self):
        self.memory_label.text = f"Memory: {psutil.virtual_memory().percent}%"

    async def export_video(self):
        if self.samples is None:
            return

        # Simple export for now
        out_path = 'output.mp4'

        self.status_label.text = "⏳ Exporting video..."
        await asyncio.sleep(0.1)


        # Render frames and pipe to ffmpeg for encoding with audio
        fps = 30

        # Get the matplotlib canvas
        fig = getattr(self.plot, 'figure', getattr(self.plot, 'fig', None))
        if fig is None:
            self.status_label.text = "❌ Failed to access plot figure"
            return
        canvas = fig.canvas
        width, height = canvas.get_width_height()

        video_stream = ffmpeg.input('pipe:0', format='rawvideo', pix_fmt='rgb24', s=f'{width}x{height}', r=fps)
        audio_stream = ffmpeg.input(self.audio_path)

        process = (
            ffmpeg
            .output(video_stream, audio_stream, out_path, pix_fmt='yuv420p', vcodec='libx264', acodec='aac', r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True, cmd=self.ffmpeg_cmd)
        )

        t = 0.0
        dt = 1.0 / fps
        try:
            while t < self.duration:
                self.update_plot(t)
                # draw and grab RGB bytes
                canvas.draw()
                frame = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                frame = frame.reshape((height, width, 3))
                process.stdin.write(frame.tobytes())
                t += dt
            process.stdin.close()
            process.wait()
        except Exception as ex:
            try:
                process.stdin.close()
            except Exception:
                pass
            self.status_label.text = f"❌ Export failed: {ex}"
            return

        self.last_export_path = out_path
        self.status_label.text = f"✓ Exported: {out_path}"

    def download_last_export(self):
        if not self.last_export_path or not os.path.exists(self.last_export_path):
            if self.status_label is not None:
                self.status_label.text = "❌ No exported video found"
            return
        ui.download(self.last_export_path)

    async def export_and_download(self):
        await self.export_video()
        self.download_last_export()

@ui.page('/')
def create_ui():
    visualizer = AudioVisualizer()

    with ui.card().style('max-width: 1200px; margin: auto;'):
        ui.label('🎵 Audio Visualizer').style('font-size: 24px; font-weight: bold; color: #00d4ff; text-align: center;')
        ui.label('The visualizer opens in a new window after both audio and image are uploaded.').style('color: #94a3b8; text-align: center;')

        # File uploads
        with ui.row():
            ui.upload(
                label='📁 Upload Audio',
                on_upload=visualizer.on_audio_upload,
                multiple=False,
                auto_upload=True,
                max_file_size=200_000_000,
                on_rejected=lambda e: visualizer.status_label.set_text('❌ Audio upload rejected (file too large or invalid)')
            ).props('accept=audio/*')
            ui.upload(
                label='🖼️ Upload Image',
                on_upload=visualizer.on_image_upload,
                multiple=False,
                auto_upload=True,
                max_file_size=50_000_000,
                on_rejected=lambda e: visualizer.status_label.set_text('❌ Image upload rejected (file too large or invalid)')
            ).props('accept=image/*')

        # Loading indicator
        visualizer.loading_indicator = ui.spinner(size='lg')
        visualizer.loading_indicator.classes('hidden')

        # Status
        visualizer.status_label = ui.label("Ready to upload audio").style('color: #4ade80; font-weight: 500;')
        visualizer.see_episode_button = ui.button('see your episode', on_click=lambda: ui.open('/visualizer'))
        visualizer.see_episode_button.disable()


@ui.page('/visualizer')
def visualizer_page():
    if CURRENT_VISUALIZER is None or not (CURRENT_VISUALIZER.audio_ready and CURRENT_VISUALIZER.image_ready):
        ui.label('No visualizer data yet. Please upload audio and an image on the main page.').style('color: #94a3b8;')
        return

    with ui.card().style('max-width: 1200px; margin: auto;'):
        CURRENT_VISUALIZER.build_visualizer_ui()
        try:
            CURRENT_VISUALIZER.prepare_visualization(open_window=False)
        except Exception:
            pass

    # Auto-start playback when the visualizer opens
    try:
        ui.run_javascript('document.getElementById("audio")?.play();')
        CURRENT_VISUALIZER.is_playing = True
        CURRENT_VISUALIZER.start_playback_timer()
    except Exception:
        pass

if __name__ in ("__main__", "__mp_main__"):
    ui.run(port=8080, title='Audio Visualizer')