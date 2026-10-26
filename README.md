# Deployment of Deep Learning Architectures in Docker Containers

This project demonstrates how to deploy a deep learning image-classification application using Docker and Docker Compose. The graphical interface is implemented with **Gradio**, while a separate **FastAPI** service performs inference using a ResNet-18 model.

The application supports two classification models: a general-purpose model pretrained on ImageNet and a specialised model fine-tuned to distinguish two butterfly species, the Red Admiral (*Vanessa atalanta*) and the Painted Lady (*Vanessa cardui*), using images from the [Leeds Butterfly Dataset](https://www.josiahwang.com/dataset/leedsbutterfly). Users can select the model and inference device (CPU or GPU). An optional training service allows the butterfly classifier to be fine-tuned using an NVIDIA GPU.

## Project Structure

```text
.
├── compose.yaml              # Configures the application services
├── prepare_data.py           # Prepares and splits the butterfly dataset
├── models/                   # Model checkpoints
├── inference/
│   ├── Dockerfile            # Builds the inference image
│   ├── requirements.txt      # Inference dependencies
│   └── app.py                # FastAPI application and inference logic
├── interface/
│   ├── Dockerfile            # Builds the interface image
│   ├── requirements.txt      # Interface dependencies
│   └── app.py                # Gradio web interface
└── training/
    ├── Dockerfile            # Builds the training image
    └── train.py              # Fine-tunes ResNet-18
```

## Requirements

- Docker Desktop with Docker Compose.
- NVIDIA GPU support configured in Docker Desktop and WSL 2 for GPU inference and training.
- Python installed on the host to prepare the dataset.
- Butterfly images organised into the two class folders described below.

## Running the Application

Run all commands from the project root, where `compose.yaml` is located.

### 1. Validate the Compose Configuration

```bash
docker compose config
```

### 2. Build the Inference and Interface Images

```bash
docker compose build inference interface
```

The inference image contains FastAPI, PyTorch, and the model inference dependencies. The interface image contains Gradio and its dependencies, without a separate PyTorch installation.

### 3. Start the Application

```bash
docker compose up -d
```

This starts the `inference` and `interface` services in the background. The training service is optional and does not start during normal application startup.

### 4. Check the Services and View Logs

```bash
docker compose ps
docker compose logs -f inference interface
```

Open the Gradio interface at [http://localhost:7860](http://localhost:7860).

The inference API is available on port `8000`, with `/health` for health checks and `/predict` for classification requests.

To stop the application, run:

```bash
docker compose down
```

The named `model-cache` volume is retained, allowing cached model weights to be reused on subsequent runs.

## Preparing the Butterfly Dataset

Before training, organise the source images into two class folders:

```text
butterfly_data/
├── admiral/
│   ├── image001.jpg
│   └── ...
└── painted_lady/
    ├── image001.png
    └── ...
```

Run the preparation script from the project root:

```bash
python prepare_data.py --source butterfly_data --output dataset
```

The script splits the source images into training, validation, and test sets. The resulting directory structure must follow the PyTorch `ImageFolder` format:

```text
dataset/
├── train/
│   ├── admiral/
│   └── painted_lady/
├── val/
│   ├── admiral/
│   └── painted_lady/
└── test/
    ├── admiral/
    └── painted_lady/
```

The training container mounts `./dataset` at `/data` as a read-only directory. Therefore, the training script expects the following paths inside the container:

- `/data/train`
- `/data/val`
- `/data/test`

Each split must contain a separate subdirectory for each butterfly class.

## Training the Butterfly Classifier

Training is performed separately from the main application. First, prepare the dataset as described above.

Build the inference image, which provides the Python and PyTorch environment used by the training image:

```bash
docker compose build inference
```

Build the training image:

```bash
docker compose --profile training build training
```

Run the training task:

```bash
docker compose --profile training run --rm training
```

The training parameters are configured through environment variables in `compose.yaml`. The default configuration uses:

- **Epochs:** 5
- **Batch size:** 8
- **Image size:** 160 pixels
- **Random seed:** 42
- **Data-loading workers:** 0

The model is evaluated on the validation set during training. The best checkpoint is saved to:

```text
models/butterfly_resnet18.pth
```

The `./models:/models` volume makes the checkpoint available on the host after the training container exits. The `--rm` option removes the completed container but preserves the checkpoint.

To use the specialised classifier in the web interface, ensure that the checkpoint exists and that the inference service is configured with:

```text
BUTTERFLY_CHECKPOINT=/models/butterfly_resnet18.pth
```

## Notes

- The Compose configuration reserves NVIDIA GPU access for the inference and training services. GPU support must be available in Docker Desktop and WSL 2.
- Selecting CPU in the interface does not override the GPU reservation configured for the inference service.
- The specialised classifier achieved 100% accuracy on a small test set. Further evaluation on a larger independent dataset is needed to assess its generalisation to unseen butterfly images.