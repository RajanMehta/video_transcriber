# Video Transcriber

This project transcribes video files and performs speaker diarization to generate a formatted transcript.

## Prerequisites

1. **Python 3.10**: This project requires Python 3.10.
2. **Poetry**: A Python dependency manager.
3. **Hugging Face Token**: You need an access token to download models. Ensure you accept the model terms & conditions on [huggingface.co](https://huggingface.co).
4. **Input Video**: A file named `video.mp4` in the project root.

## Setup Instructions

1. Let poetry know that it should expect a virtual environment within the project directory
```bash
poetry config virtualenvs.in-project true
``` 

2. Create a virtual env with name `.venv` as poetry tries to find that name by default.
```bash
python3.10 -m venv .venv
```

3. Activate the environment
```bash
source .venv/bin/activate
```

4. Install packages
```bash
poetry install
```

5. You should now be able to run the main.py file
```bash
python video_transcriber.py
```

## Usage

Run the main script to process `video.mp4` and generate the transcript:
```bash
python video_transcriber.py
```
