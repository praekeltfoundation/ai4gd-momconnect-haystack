# ai4gd-momconnect-haystack
Haystack pipelines for use on the AI4GD study being run on MomConnect

## Environment setup and running the application (using Docker):
*In the root directory*:
1. Create a `.env` file containing an OpenAI API environment variable: `OPENAI_API_KEY=<your API key>`
2. Open a terminal and run `docker-compose build`.
3. Then run `docker-compose run python-app`

## Environment setup and running the application (locally)
*In the root directory*:
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Create a `.env` file containing an OpenAI API environment variable: `OPENAI_API_KEY=<your API key>`
3. Open a terminal and run `uv run main`.
