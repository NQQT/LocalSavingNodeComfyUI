import torch
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import datetime
# Import ComfyUI's PromptServer for client-server communication
# This allows us to send data from Kaggle (server) to the browser (client)
from server import PromptServer

class ImageSaveNode:
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