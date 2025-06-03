# ai4gd-momconnect-haystack
Haystack pipelines for use on the AI4GD study being run on MomConnect

## Environment setup and running the application (using Docker):
*In the root directory*:
1. Create a `.env` file containing an OpenAI API environment variable: `OPENAI_API_KEY=<your API key>` and an API token `API_TOKEN=<generate-random-characters>`
2. Open a terminal and run `docker-compose build`.
3. Then run `docker-compose run python-app`

## Environment setup and running the application (locally)
*In the root directory*:
1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Create a `.env` file containing an OpenAI API environment variable: `OPENAI_API_KEY=<your API key>`
3. Open a terminal and run `uv run main`.

## Configuration
The following environment variables control the application configuration:
OPENAI_API_KEY - (required) the API access key for OpenAI
API_TOKEN - (required) the token that is to be supplied when accessing the API.
WEAVIATE_URL - (optional) if supplied, the URL to the weaviate instance. If not supplied, an embedded instance will be used
WEAVIATE_API_KEY - (optional) if supplied, the API key to use to connect to the weaviate instance. If not supplied, no authentication is provided when connecting
