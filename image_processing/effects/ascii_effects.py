import logging
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from image_processing.core.image_processor import BaseImageProcessor

from ..core.effect_processor import EffectProcessor
from ..core.utils import ImageUtils

class ASCIIProcessor(BaseImageProcessor):
    """
    Handles creation and management of image effect animations.
    """

    def __init__(self, image_input: Union[str, bytes, Image.Image, BytesIO]):
        """
        Initialize animation processor.

        Args:
            image_input: Source image in various formats
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("AnimationProcessor")

        # Load and validate input image
        self.base_image = ImageUtils.load_image(image_input)

        print("Loaded image")
        # Ensure dimensions are even for video encoding
        width, height = self.base_image.size
        new_width, new_height = ImageUtils.ensure_size_even(width, height)

        if (new_width, new_height) != (width, height):
            new_image = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            new_image.paste(
                self.base_image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            self.base_image = new_image

        print("Evened dimensions")

        # Set up temporary directory for frames
        self.temp_dir = Path(tempfile.mkdtemp(prefix="anim_frames_"))
        self.frames_dir = self.temp_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)

        # Initialize processors
        self.font = ImageFont.load_default()

    def get_char_size(self, char: str = 'A') -> tuple[int, int]:
        """Get character dimensions using getbbox"""
        bbox = self.font.getbbox(char)
        if bbox is None:
            return (10, 10)  # Fallback default size
        return (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))

    def get_optimal_dimensions(self, target_width: int = 640) -> tuple[int, float]:
        """Calculate optimal columns and scale for ASCII art"""
        # For typical monospace fonts, each character is about half as tall as it is wide
        # Standard terminal aspect ratio is roughly 2:1 (width:height)
        char_width, char_height = self.get_char_size('A')
        aspect_ratio = char_width / char_height

        # Calculate optimal number of columns based on target width
        cols = int(target_width / char_width)

        # Calculate scale factor to maintain aspect ratio
        # Most ASCII art looks best with scale around 0.43-0.47
        scale = 0.45 * aspect_ratio

        return cols, scale

    def convert_to_ascii(
                        self, image: Image.Image,
                        cols: int = 150,
                        scale: Optional[float] = None,
                        more_levels: bool = False
    ) -> List[str]:
        # Convert to grayscale
        img = image.convert('L')
    
        # Simple resize maintaining aspect ratio
        W, H = img.size
        aspect_ratio = H / W
        rows = int(cols * aspect_ratio * 0.55)  # Compensate for character spacing

        img = img.resize((cols, rows), Image.Resampling.LANCZOS)
        pixels = np.array(img)

        # Simpler character set, ordered from darkest to lightest
        if more_levels:
            ramp = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,"^`\''
        else:
            ramp = '@#H&9$2?-. '

        result = []
        for row in range(rows):
            line = ""
            for col in range(cols):
                pixel_value = pixels[row, col]
                # Simple linear mapping
                char_idx = int((pixel_value / 255.0) * (len(ramp) - 1))
                # Single character (no doubling)
                line += ramp[char_idx]
            result.append(line)
        
        return result
    
    def ascii_to_image(self, ascii_art: List[str], background_color=(0, 0, 0), text_color=(255, 255, 255)) -> Image.Image:
        if not ascii_art:
            raise ValueError("ASCII art list is empty")
        
        # Use default font
        self.font = ImageFont.load_default()
        
        char_width, char_height = self.get_char_size('A')
        padding = 20
        line_spacing = int(char_height * 1.2)
        
        # Calculate dimensions
        max_line_length = max(len(line) for line in ascii_art)
        image_width = max_line_length * char_width + padding * 2
        image_height = len(ascii_art) * line_spacing + padding * 2
        
        image = Image.new('RGB', (image_width, image_height), background_color)
        draw = ImageDraw.Draw(image)
        
        y = padding
        for line in ascii_art:
            draw.text((padding, y), line, fill=text_color, font=self.font)
            y += line_spacing
        
        return image

    def convert_to_ascii_image(self, image: Image.Image, cols: int = 80, scale: float = 0.43, 
                           moreLevels: bool = False,
                           background_color=(0, 0, 0),
                           text_color=(255, 255, 255)) -> Image.Image:
        """Convert input image to ASCII art to a new image in one step"""
        print("Starting conversion process...")
        print(f"Input image mode: {image.mode}")
        print(f"Input image size: {image.size}")
    
        # Convert to grayscale and check values
        gray_image = image.convert('L')
        pixels = np.array(gray_image)
        print(f"Grayscale range: min={pixels.min()}, max={pixels.max()}")
        
        # Generate ASCII art
        ascii_art = self.convert_to_ascii(image, cols, scale, moreLevels)
        print(f"ASCII art generated, number of lines: {len(ascii_art)}")
        if ascii_art:
            print(f"Sample line (first line): {ascii_art[0][:30]}...")
        
        # Convert back to image
        result = self.ascii_to_image(ascii_art, background_color, text_color)
        print(f"Final image size: {result.size}")
        return result

    def generate_frames(
        self, effects: List[Tuple[str, Dict[str, Any]]], num_frames: int = 30
    ) -> List[Path]:
        """
        Generate animation frames with multiple effects.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate

        Returns:
            List of paths to generated frame files
        """
        frame_paths = []

        try:
            for i in range(num_frames):
                # Calculate animation progress
                progress = i / (num_frames - 1)

                # Process frame with interpolated parameters
                frame = self.base_image.copy()
                processor = EffectProcessor(frame)

                for effect_name, params in effects:
                    # Interpolate parameters
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    # Apply effect
                    processor.apply_effect(effect_name, frame_params)

                # Save frame
                frame_path = self.frames_dir / f"frame_{i:04d}.png"
                processor.save(frame_path)
                frame_paths.append(frame_path)

                self.logger.info(f"Generated frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"Frame generation error: {str(e)}")
            raise

        return frame_paths

    def create_video(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        frame_rate: int = 24,
        crf: int = 23,
        preset: str = "medium",
    ) -> Optional[Path]:
        """
        Create video from frames using ffmpeg.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output video file
            frame_rate: Frames per second
            crf: Constant Rate Factor (18-28 recommended)
            preset: ffmpeg encoding preset

        Returns:
            Path to output video file
        """
        if not frame_paths:
            raise ValueError("No frames provided for video creation")

        if not output_path:
            output_path = self.temp_dir / "output.mp4"
        else:
            output_path = Path(output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Construct ffmpeg command
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(frame_rate),
                "-i",
                str(self.frames_dir / "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                "-vf",
                "format=yuv420p",
                str(output_path),
            ]

            # Run ffmpeg
            result = subprocess.run(
                ffmpeg_cmd, check=True, capture_output=True, text=True
            )

            if result.returncode == 0:
                return output_path

        except subprocess.CalledProcessError as e:
            self.logger.error(f"ffmpeg error: {e.stderr}")
        except Exception as e:
            self.logger.error(f"Video creation error: {str(e)}")

        return None

    def create_gif(
        self,
        frame_paths: List[Path],
        output_path: Optional[Union[str, Path]] = None,
        duration: int = 50,
    ) -> Optional[Path]:
        """
        Create animated GIF from frames.

        Args:
            frame_paths: List of frame file paths
            output_path: Path for output GIF file
            duration: Frame duration in milliseconds

        Returns:
            Path to output GIF file
        """
        if not frame_paths:
            raise ValueError("No frames provided for GIF creation")

        if not output_path:
            output_path = self.temp_dir / "output.gif"
        else:
            output_path = Path(output_path)

        try:
            # Load frames and optimize for GIF
            frames = []
            for frame_path in frame_paths:
                with Image.open(frame_path) as frame:
                    # Convert to P mode with adaptive palette
                    if frame.mode != "P":
                        frame = frame.convert("RGBA").convert(
                            "P", palette=Image.Palette.ADAPTIVE, colors=256
                        )
                    frames.append(frame.copy())

            # Save as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=duration,
                loop=0,
                optimize=True,
            )

            return output_path

        except Exception as e:
            self.logger.error(f"GIF creation error: {str(e)}")
            return None

    def create_ascii_animation(
        self,
        effects: Dict[str, Dict[str, Any]],
        num_frames: int = 30,
        cols: int = 120,
        scale: float = 0.43,
    ) -> List[str]:
        """
        Create ASCII art animation frames.

        Args:
            effects: List of (effect_name, parameters) tuples
            num_frames: Number of frames to generate
            cols: Number of columns for ASCII art
            scale: Character aspect ratio adjustment

        Returns:
            List of ASCII art strings
        """
        ascii_frames = []

        try:
            for i in range(num_frames):
                progress = i / (num_frames - 1)

                # Process frame with effects
                self.current_image = self.base_image.copy()
                processor = EffectProcessor(self.current_image)

                for effect_name, params in effects.items():
                    frame_params = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, tuple) and len(param_value) == 2:
                            frame_params[param_name] = ImageUtils.interpolate_value(
                                param_value[0], param_value[1], progress
                            )
                        else:
                            frame_params[param_name] = param_value

                    processor.apply_effect(effect_name, frame_params)

                self.current_image = processor.current_image.copy()

                # Convert to ASCII
                ascii_frame = self.convert_to_ascii(self.current_image, cols=cols, scale=scale)
                ascii_frames.append("\n".join(ascii_frame))

                self.logger.info(f"Generated ASCII frame {i + 1}/{num_frames}")

        except Exception as e:
            self.logger.error(f"ASCII animation error: {str(e)}")
            raise

        return ascii_frames

    def cleanup(self):
        """Clean up temporary files."""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            self.logger.error(f"Cleanup error: {str(e)}")

    def __del__(self):
        """Ensure cleanup on object destruction."""
        self.cleanup()
