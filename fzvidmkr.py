#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
import subprocess
import tempfile
import shutil
import argparse
from pathlib import Path
from PIL import Image
import struct
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
import time
import ffmpeg

BUNDLE_SIGNATURE = b"BND!VID"
VERSION = 1

class FlipperVideoConverterAlt:
    def __init__(self, input_video, output_file, width=128, height=64, fps=30, sample_rate=44100, source_width=None, source_height=None):
        self.input_video = input_video
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.sample_rate = sample_rate
        self.source_width = source_width
        self.source_height = source_height
        self.temp_dir = None
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'total_audio_size': 0,
            'audio_chunk_size': sample_rate // fps,
            'estimated_size': 0
        }
        self.console = Console()

    def create_temp_directory(self):
        self.temp_dir = tempfile.mkdtemp(prefix="flipper_video_alt_")
        os.makedirs(os.path.join(self.temp_dir, "frames"), exist_ok=True)
        return self.temp_dir

    def get_video_dimensions(self):
        if self.source_width is not None and self.source_height is not None:
            return self.source_width, self.source_height
        
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0", self.input_video
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            try:
                width = int(lines[0].strip())
                height = int(lines[1].strip())
                return width, height
            except (ValueError, IndexError):
                pass
        
        # Fallback
        return 1920, 1080

    def extract_frames(self):
        frames_path = os.path.join(self.temp_dir, "frames")
        frame_pattern = os.path.join(frames_path, "frame%07d.bmp")
        
        try:
            probe = ffmpeg.probe(self.input_video)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if video_stream is not None:
                total_duration = float(video_stream.get('duration', 0))
                avg_frame_rate = video_stream.get('avg_frame_rate', '30/1')
                if avg_frame_rate and avg_frame_rate != 'N/A':
                    try:
                        num, den = avg_frame_rate.split('/')
                        fps_from_video = int(num) / int(den)
                        total_frames = int(total_duration * fps_from_video)
                    except (ValueError, ZeroDivisionError):
                        total_frames = 1000  # fallback value
                else:
                    total_frames = 1000  # fallback value
            else:
                total_frames = 1000  # fallback value
        except Exception:
            total_frames = 1000  # fallback value
        
        self.stats['total_frames'] = total_frames
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Extracting frames with threshold filter...", total=total_frames)
            
            try:
                stream = ffmpeg.input(self.input_video)
                stream = ffmpeg.filter(stream, 'scale', self.width, self.height)
                stream = ffmpeg.filter(stream, 'threshold')
                stream = ffmpeg.output(stream, frame_pattern, r=self.fps)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
            except ffmpeg.Error as e:
                self.console.print(f"[red]FFmpeg frame extraction failed:[/red] {e.stderr.decode() if e.stderr else str(e)}")
                try:
                    stream = ffmpeg.input(self.input_video)
                    stream = ffmpeg.filter(stream, 'scale', self.width, self.height)
                    stream = ffmpeg.filter(stream, 'format', 'rgb24')
                    stream = ffmpeg.filter(stream, 'threshold')
                    stream = ffmpeg.output(stream, frame_pattern, r=self.fps)
                    ffmpeg.run(stream, overwrite_output=True, quiet=True)
                except ffmpeg.Error as e2:
                    self.console.print(f"[yellow]Trying alternative FFmpeg command without threshold...[/yellow]")
                    try:
                        stream = ffmpeg.input(self.input_video)
                        stream = ffmpeg.filter(stream, 'scale', self.width, self.height)
                        stream = ffmpeg.output(stream, frame_pattern, r=self.fps)
                        ffmpeg.run(stream, overwrite_output=True, quiet=True)
                    except ffmpeg.Error as e3:
                        raise Exception(f"FFmpeg frame extraction failed: {e3.stderr.decode() if e3.stderr else str(e3)}")
            
            actual_frames = len([f for f in os.listdir(frames_path) if f.endswith('.bmp')])
            progress.update(task, completed=actual_frames, total=actual_frames)
        
        actual_frames = len([f for f in os.listdir(frames_path) if f.endswith('.bmp')])
        self.stats['total_frames'] = actual_frames
        return actual_frames

    def extract_audio(self):
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        
        try:
            probe = ffmpeg.probe(self.input_video)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            
            if not audio_streams:
                self.console.print(f"[yellow]No audio stream found, creating empty audio...[/yellow]")

                video_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'video']
                if video_streams:
                    duration = float(video_streams[0].get('duration', 3))
                else:
                    duration = 3
                
                cmd = [
                    "ffmpeg",
                    "-f", "lavfi",
                    "-i", f"anullsrc=r={self.sample_rate}:d={duration}",
                    "-acodec", "pcm_u8",
                    "-ac", "1",
                    "-y",
                    audio_path
                ]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if result.returncode != 0:
                    raise Exception(f"Silent audio generation failed: {result.stderr.decode()}")
            else:
                self.console.print(f"[yellow]Extracting audio at {self.sample_rate}Hz...[/yellow]")
                
                stream = ffmpeg.input(self.input_video)
                stream = ffmpeg.output(stream, audio_path, ac=1, acodec='pcm_u8', ar=self.sample_rate)
                ffmpeg.run(stream, overwrite_output=True, quiet=True)
        except ffmpeg.Error as e:
            raise Exception(f"FFmpeg audio extraction failed: {e.stderr.decode() if e.stderr else str(e)}")
        
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
            if audio_stream is not None:
                duration = float(audio_stream.get('duration', 0))
                self.stats['total_audio_size'] = int(duration * self.sample_rate)
            else:
                file_size = os.path.getsize(audio_path)
                audio_data_size = max(0, file_size - 44)
                self.stats['total_audio_size'] = audio_data_size
        except Exception:
            self.stats['total_audio_size'] = 0
        
        return audio_path

    def find_audio_data_start(self, audio_file):
        with open(audio_file, 'rb') as f:
            pos = 0
            while True:
                f.seek(pos)
                sig = f.read(4)
                    pos += 8
                    break
                pos += 1
                if pos > 1000:
                    break
        return pos

    def bmp_load_and_convert(self, filepath):
        img = Image.open(filepath)
        if img.mode == 'BGR':
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        if img.size != (self.width, self.height):
            img = img.resize((self.width, self.height), Image.Resampling.LANCZOS)
        
        img_data = np.array(img)

        binary_frame = []
        
        total_pixels = self.width * self.height
        flat_img_data = img_data.reshape(total_pixels, 3)
        for j in range(0, total_pixels, 8):
            byte_val = 0
            for k in range(8):  # 8 bits in a byte
                pixel_idx = j + k
                if pixel_idx < total_pixels:
                    pixel_r = flat_img_data[pixel_idx][0]
                    if pixel_r > 128:
                        bit_val = 0
                    else:
                        bit_val = 1
                    
                    byte_val |= (bit_val << k)
            
            binary_frame.append(byte_val)
        
        return binary_frame

    def create_bnd_file(self):
        frames_dir = os.path.join(self.temp_dir, "frames")
        audio_file = os.path.join(self.temp_dir, "audio.wav")
        
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.bmp')])
        num_frames = len(frame_files)
        self.stats['total_frames'] = num_frames
        
        frame_size = self.width * self.height // 8
        audio_chunk_size = self.sample_rate // self.fps
        self.stats['estimated_size'] = (
            len(BUNDLE_SIGNATURE) - 1 +  # signature
            1 +  # version
            4 +  # num_frames
            2 +  # audio_chunk_size
            2 +  # sample_rate
            1 +  # height
            1 +  # width
            (frame_size + audio_chunk_size) * num_frames  # data
        )
        
        audio_data_start = self.find_audio_data_start(audio_file)
        
        with open(self.output_file, 'wb') as bundle, open(audio_file, 'rb') as audio:
            audio.seek(audio_data_start)

            bundle.write(BUNDLE_SIGNATURE[:7])
            bundle.write(struct.pack('B', VERSION))
            bundle.write(struct.pack('I', num_frames))
            bundle.write(struct.pack('H', self.sample_rate // self.fps))
            bundle.write(struct.pack('H', self.sample_rate))
            bundle.write(struct.pack('B', self.height))
            bundle.write(struct.pack('B', self.width))
            
            self.stats['processed_frames'] = 0
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=self.console
            ) as progress:
                task = progress.add_task("Creating BND file...", total=num_frames)
                
                for i, frame_file in enumerate(frame_files):
                    frame_path = os.path.join(frames_dir, frame_file)
                    
                    binary_frame = self.bmp_load_and_convert(frame_path)
                    
                    bundle.write(bytes(binary_frame))


                    audio_chunk = audio.read(self.sample_rate // self.fps)
                    bundle.write(audio_chunk)
                    
                    self.stats['processed_frames'] += 1
                    progress.update(task, advance=1)

    def show_stats(self):
        table = Table(title="Conversion Statistics (Makefile Project Version)")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", justify="right", style="magenta")
        
        table.add_row("Input Video", os.path.basename(self.input_video))
        table.add_row("Output File", os.path.basename(self.output_file))
        table.add_row("Output Dimensions", f"{self.width}x{self.height}")
        table.add_row("Target FPS", str(self.fps))
        table.add_row("Audio Sample Rate", f"{self.sample_rate}Hz")
        table.add_row("Audio Chunk Size", f"{self.sample_rate // self.fps} bytes")
        table.add_row("Total Frames", str(self.stats['total_frames']))
        table.add_row("Processed Frames", str(self.stats['processed_frames']))
        table.add_row("Estimated File Size", f"{self.stats['estimated_size']} bytes")
        table.add_row("Source Dimensions", f"{self.source_width or 'Auto'}x{self.source_height or 'Auto'}")
        
        self.console.print(table)

    def clean_up(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def convert(self):
        try:
            self.console.print(Panel("Flipper Zero Video Converter (Makefile Project Version)", expand=False, border_style="green"))
            self.console.print(f"[bold]Input:[/bold] {self.input_video}")
            self.console.print(f"[bold]Output:[/bold] {self.output_file}\n")
            
            self.create_temp_directory()
            self.console.print(f"[green]Created temporary directory:[/green] {self.temp_dir}\n")
            
            self.console.print("[bold blue]Step 1/3:[/bold blue] Extracting frames with threshold filter...")
            frame_count = self.extract_frames()
            self.console.print(f"[green]Extracted {frame_count} frames[/green]\n")
            
            self.console.print("[bold blue]Step 2/3:[/bold blue] Extracting audio...")
            audio_path = self.extract_audio()
            self.console.print(f"[green]Extracted audio file[/green]\n")
            
            self.console.print("[bold blue]Step 3/3:[/bold blue] Creating BND file...")
            self.create_bnd_file()
            self.console.print(f"[green]BND file created successfully:[/green] {self.output_file}\n")
            
            self.show_stats()
            
            self.clean_up()
            self.console.print(f"\n[bold green]Conversion completed successfully![/bold green]")
            
        except Exception as e:
            self.console.print(f"[red]Error during conversion:[/red] {str(e)}")
            self.clean_up()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MP4 files to Flipper Zero BND format (Makefile Project version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i input.mp4 -o output.bnd
  %(prog)s -i video.mp4 -o video.bnd -w 96 -ht 64 -f 30 -sr 44100 -sw 1920 -sh 1080
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input MP4 video file')
    parser.add_argument('-o', '--output', required=True, help='Output BND file')
    parser.add_argument('-w', '--width', type=int, default=128, help='Width of output video (max 128, multiple of 8)')
    parser.add_argument('-ht', '--height', type=int, default=64, help='Height of output video (max 64)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Frames per second (recommended <= 30)')
    parser.add_argument('-sr', '--sample-rate', type=int, default=44100, help='Audio sample rate (use integer that divides evenly by FPS)')
    parser.add_argument('-sw', '--source-width', type=int, help='Width of source video (for FFmpeg command)')
    parser.add_argument('-sh', '--source-height', type=int, help='Height of source video (for FFmpeg command)')
    
    args = parser.parse_args()
    
    if args.width > 128 or args.height > 64:
        print("Error: Width must not exceed 128 and height must not exceed 64")
        sys.exit(1)
    
    if args.width % 8 != 0:
        print("Error: Width must be a multiple of 8")
        sys.exit(1)
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        sys.exit(1)
    
    try:
        import ffmpeg
        probe_result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if probe_result.returncode != 0:
            print("Error: FFmpeg is required but not found. Please install FFmpeg.")
            sys.exit(1)
    except (FileNotFoundError, ImportError):
        print("Error: FFmpeg and/or ffmpeg-python are required but not found. Please install both.")
        sys.exit(1)
    
    converter = FlipperVideoConverterAlt(
        input_video=args.input,
        output_file=args.output,
        width=args.width,
        height=args.height,
        fps=args.fps,
        sample_rate=args.sample_rate,
        source_width=args.source_width,
        source_height=args.source_height
    )
    
    converter.convert()


if __name__ == "__main__":
    main()
