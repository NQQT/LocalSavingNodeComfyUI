import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import datetime
import subprocess
from typing import List, Union

# Import ComfyUI's PromptServer for client-server communication
# This allows us to send data from Kaggle (server) to the browser (client)
from server import PromptServer

class KaggleLocalSaveNode:
    """
    Custom ComfyUI node that saves generated images directly to the local PC
    instead of Kaggle's cloud output folder.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "prefix": ("STRING", {"default": "kaggle_generated"}),
                "file_format": (["PNG", "JPEG"], {"default": "PNG"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_images"
    CATEGORY = "image"
    OUTPUT_NODE = True

    def process_images(self, images, prefix, file_format):
        try:
            # Generate timestamp for unique filenames
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # List to store image data for download
            image_data_list = []

            # Process each image in the batch
            for i in range(len(images)):
                # Extract image tensor
                img_tensor = images[i]

                # Convert tensor to numpy array
                img_numpy = 255. * img_tensor.cpu().numpy()

                # Convert numpy array to PIL Image
                img = Image.fromarray(np.clip(img_numpy, 0, 255).astype(np.uint8))

                # Create filename with timestamp and index
                filename = f"{prefix}_{timestamp}_{i + 1:03d}"
                full_filename = f"{filename}.{file_format.lower()}"

                # Create buffer for image data
                buffered = BytesIO()

                # Save image to buffer in specified format
                if file_format == "PNG":
                    img.save(buffered, format="PNG")
                else:
                    img.save(buffered, format="JPEG", quality=95)

                # Convert image to base64 string for browser download
                img_str = base64.b64encode(buffered.getvalue()).decode()

                # Add image data to list
                image_data_list.append({
                    "filename": full_filename,
                    "data": img_str,
                    "format": file_format.lower()
                })

            # Send data to client for download
            PromptServer.instance.send_sync("kaggle_local_save_data", {
                "images": image_data_list
            })

            # Return original images to allow continued workflow
            return (images,)

        except Exception as e:
            error_msg = f"Error processing images: {str(e)}"
            PromptServer.instance.send_sync("kaggle_local_save_error", {
                "message": error_msg
            })
            raise Exception(error_msg)

class KaggleLocalSaveVideoNode:
    """
    Custom ComfyUI node that saves generated images directly to the local PC
    instead of Kaggle's cloud output folder.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "prefix": ("STRING", {"default": "kaggle_generated"}),
                "file_format": (["WEBM", "MP4"], {"default": "WEBM"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_video"
    CATEGORY = "image"
    OUTPUT_NODE = True


    def images_to_video_buffer(
            images: Union[List[torch.Tensor], torch.Tensor],
            frame_rate: float = 24.0,
            format: str = "mp4",  # "mp4" or "webm"
            quality: str = "medium"  # "low", "medium", "high"
    ) -> bytes:

        # Handle different input formats
        if isinstance(images, torch.Tensor):
            if len(images.shape) == 4:  # Batched [B,H,W,C]
                images = [images[i] for i in range(images.shape[0])]
            else:
                images = [images]

        if len(images) == 0:
            raise ValueError("No images provided")

        # Get first image to determine dimensions
        first_img = images[0]
        if len(first_img.shape) == 3:  # [H,W,C]
            height, width, channels = first_img.shape
        else:
            raise ValueError("Expected image tensor shape [H,W,C]")

        # Quality presets
        quality_settings = {
            "low": {"crf": "28", "preset": "fast"},
            "medium": {"crf": "23", "preset": "medium"},
            "high": {"crf": "18", "preset": "slow"}
        }

        settings = quality_settings.get(quality, quality_settings["medium"])

        # Format-specific encoding args
        if format.lower() == "webm":
            codec_args = [
                "-c:v", "libvpx-vp9",
                "-crf", settings["crf"],
                "-b:v", "0",  # Use CRF mode
                "-row-mt", "1"
            ]
            container = "webm"
        else:  # mp4
            codec_args = [
                "-c:v", "libx264",
                "-preset", settings["preset"],
                "-crf", settings["crf"],
                "-pix_fmt", "yuv420p"
            ]
            container = "mp4"

        # Build ffmpeg command
        ffmpeg_cmd = [
                         "ffmpeg",
                         "-f", "rawvideo",
                         "-pix_fmt", "rgb24",
                         "-s", f"{width}x{height}",
                         "-r", str(frame_rate),
                         "-i", "-",  # Input from stdin
                         "-y",  # Overwrite output
                     ] + codec_args + [
                         "-f", container,
                         "-"  # Output to stdout
                     ]

        # Start ffmpeg process
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            # Convert and feed images to ffmpeg
            for img_tensor in images:
                # Convert tensor to numpy (same as your example)
                img_numpy = 255. * img_tensor.cpu().numpy()
                img_bytes = np.clip(img_numpy, 0, 255).astype(np.uint8)

                # Ensure RGB format
                if img_bytes.shape[-1] == 4:  # RGBA -> RGB
                    img_bytes = img_bytes[..., :3]
                elif len(img_bytes.shape) == 2:  # Grayscale -> RGB
                    img_bytes = np.stack([img_bytes] * 3, axis=-1)

                # Write raw bytes to ffmpeg
                process.stdin.write(img_bytes.tobytes())

            # Close stdin and get output
            process.stdin.close()
            stdout_data, stderr_data = process.communicate()

            if process.returncode != 0:
                error_msg = stderr_data.decode('utf-8') if stderr_data else "Unknown error"
                raise RuntimeError(f"FFmpeg failed: {error_msg}")

            return stdout_data

        except Exception as e:
            process.kill()
            process.wait()
            raise e

    def process_video(self, images, prefix, file_format):
        try:
            images_to_video_buffer(images)

            # Send data to client for download
            PromptServer.instance.send_sync("kaggle_local_save_data", {
                "images": image_data_list
            })

            # Return original images to allow continued workflow
            return (images,)

        except Exception as e:
            error_msg = f"Error processing images: {str(e)}"
            PromptServer.instance.send_sync("kaggle_local_save_error", {
                "message": error_msg
            })
            raise Exception(error_msg)



# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "KaggleLocalSaveNode": KaggleLocalSaveNode,
    "KaggleLocalSaveVideoNode":KaggleLocalSaveVideoNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KaggleLocalSaveNode": "Kaggle Local Save Image",
    "KaggleLocalSaveVideoNode": "Kaggle Local Save Video"
}