// Client-side JavaScript for handling image downloads from Kaggle ComfyUI
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

console.log("what is this: ", app);
console.log("what is really this: ", window.comfyAPI.app.app)

// Add event listener for our custom node's messages
app.registerExtension({
    name: "KaggleLocalSave",
    setup() {
        // Listen for image data from the server
        api.addEventListener("kaggle_local_save_data", (event) => {
            const data = event.detail.images;
            // Process each image for download
            data.forEach(image => {
                // Create download link
                downloadImage(image.data, image.filename, image.format);
            });
        });

        // Listen for error messages
        api.addEventListener("kaggle_local_save_error", (event) => {
            console.error("Kaggle Local Save Error:", event.detail.message);
            // Optionally display error to user
            app.ui.notifications.create("Error saving image locally: " + event.detail.message, "error", 5000);
        });
    }
});

// Function to trigger download of base64 encoded image
function downloadImage(base64Data, filename, format) {
    // Create download link
    const link = document.createElement('a');
    link.href = `data:image/${format};base64,${base64Data}`;
    link.download = filename;

    console.log('item:', link);

    // Append to document, click, and remove
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
