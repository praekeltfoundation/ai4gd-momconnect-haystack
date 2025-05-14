# ai4gd-momconnect-haystack
Haystack pipelines for use on the AI4GD study being run on MomConnect

## Environment setup (on Ubuntu/WSL):
1. Install Python (I used 3.12)
2. Install `pipx` to install the `uv` package and project manager:
  - `sudo apt install pipx`
  - `pipx install uv`
3. Make sure you are in the project's root directory and create a virtual environment using `uv`
  - `uv venv` (I _don't think_ you'll have to initialize `uv` before this, using `uv init`)
4. Install dependencies
  - `uv pip compile docs/requirements.in --universal --output-file docs/requirements.txt`
  - `uv pip sync docs/requirements.txt`
5. Add you OpenAI API key in a `.env` file in the root directory
5. Run the simulation of Onboarding + the DMA Assessment
  - `uv run main.py`
