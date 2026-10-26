import os
from io import BytesIO

import gradio as gr
import requests

API_URL = os.getenv(
    "API_URL",
    "http://inference:8000/predict",
)


def classify_image(image, selected_model, selected_device):
    """Send the image, model choice, and device choice to the API."""

    if image is None:
        raise gr.Error("Please upload an image first.")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    try:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    "image.png",
                    image_bytes,
                    "image/png",
                )
            },
            data={
                "model": selected_model,
                "device": selected_device,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()

    except requests.Timeout:
        raise gr.Error(
            "The inference server timed out. Please try again."
        )

    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)

        raise gr.Error(
            f"The inference API returned an error: {detail}"
        )

    except requests.RequestException as exc:
        raise gr.Error(
            f"Could not connect to the inference server: {exc}"
        )

    except ValueError:
        raise gr.Error(
            "The inference server returned an invalid response."
        )

    predictions = result.get("predictions", [])

    if not predictions:
        raise gr.Error(
            "No predictions were returned by the model."
        )

    # Keep Gradio's boxed result panel and confidence bars.
    scores = {}

    for item in predictions:
        if "label" not in item:
            continue

        confidence = item.get(
            "confidence",
            item.get("probability", item.get("score")),
        )

        if confidence is None:
            raise gr.Error(
                f"No confidence score for {item['label']}."
            )

        scores[str(item["label"])] = float(confidence)

    if not scores:
        raise gr.Error(
            "The inference server returned an unexpected response."
        )

    device = result.get("device", selected_device)

    # if selected_model == "butterfly":
    #     device = f"{device} · Butterfly-specific model"
    # else:
    #     device = f"{device} · Universal ImageNet model"

    return scores, device


with gr.Blocks(title="ResNet-18 Image Classifier") as demo:

    gr.Markdown(
        """
        # ResNet-18 Image Classifier

        Upload an image, select a model, and choose CPU or GPU inference.
        """
    )

    with gr.Row():

        with gr.Column():
            image_input = gr.Image(
                label="Upload an image",
                type="pil",
                sources=["upload"],
            )

            classify_button = gr.Button(
                "Classify image",
                variant="primary",
            )

        with gr.Column():
            results = gr.Label(
                label="Top-5 predictions (confidence)",
                num_top_classes=5,
            )

            device_output = gr.Textbox(
                label="Inference device",
                interactive=False,
            )

            with gr.Group():
                model_choice = gr.Radio(
                    choices=[
                        (
                            "Universal ResNet-18 · 1,000 ImageNet classes",
                            "imagenet",
                        ),
                        (
                            "Specialized model · 2 butterfly species",
                            "butterfly",
                        ),
                    ],
                    value="imagenet",
                    label="Inference model",
                )

                device_choice = gr.Radio(
                    choices=[
                        ("GPU — NVIDIA CUDA", "cuda"),
                        ("CPU", "cpu"),
                    ],
                    value="cuda",
                    label="Compute device",
                )

    classify_button.click(
        fn=classify_image,
        inputs=[
            image_input,
            model_choice,
            device_choice,
        ],
        outputs=[
            results,
            device_output,
        ],
    )

    # gr.Markdown(
    #     """
    #     **Universal model:** ImageNet, 1,000 classes.

    #     **Specialized model:** Red Admiral and Painted Lady butterflies.
    #     This model always selects between its two trained species.

    #     GPU inference requires CUDA to be available in the inference container.
    #     The API runs in a separate Docker container.
    #     """
    # )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )