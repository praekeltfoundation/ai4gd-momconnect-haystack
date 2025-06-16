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

`OPENAI_API_KEY` - (required) the API access key for OpenAI

`API_TOKEN` - (required) the token that is to be supplied when accessing the API.

`WEAVIATE_URL` - (optional) if supplied, the URL to the weaviate instance. If not supplied, an embedded instance will be used

`WEAVIATE_API_KEY` - (optional) if supplied, the API key to use to connect to the weaviate instance. If not supplied, no authentication is provided when connecting

`SENTRY_DSN` - (optional) if supplied, then we configure to send any errors to the provided DSN

## API

### Documentation
The API is documented in this readme, but there is also automatically generated documentation available at the `/docs` endpoint of the service.

### Authentication
Authentication is handled through the `Authorization` HTTP header, the value of which should be `Token <token>`. The token is configured using the `API_TOKEN` environment variable. There is only one token for the whole service.

### Onboarding
Onboarding is handled through the `/v1/onboarding` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_input`: The message that the user sent to us, as a string

`user_context`: A dictionary of contact fields that we want to send to the LLM

`chat_history`: (optional) a list of strings of the chat history up until this point (not including `user_input`)

for example:
```json
{
  "user_input": "Hello!",
  "user_context": {}
}
```

It will respond with the following fields:

`question`: The next question to ask the user, as a string. If not supplied, onboarding is complete.

`user_context`: An updated dictionary of contact fields.

`chat_history`: a list of string of the updated chat history. You can pass this directly to the next request to maintain chat history.

for example
```json
{
  "user_context": {"area_type": "City"},
  "question": "Which province are you currently living in? 🏡",
  "chat_history": [
      "User to System: Hello!",
      "System to User: Which province are you currently living in? 🏡",
  ]
}
```

### Assessment
Assessments are handled through the `/v1/assessment` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_input`: The user's message, as a string

`user_context`: The user's contact fields that we want to share with the LLM

`flow_id`: Which assessment to run, as a string. Currenty only `dma-assessment` is implemented

`question_number`: Which question number we are on, as an integer.

`chat_history`: An optional list of strings, showing the conversation history up until this point, without the latest `user_input`

for example:
```json
{
  "user_context": {},
  "user_input": "Hello!",
  "flow_id": "dma-assessment",
  "question_number": 1,
}
```

The API responds with the following fields:

`question`: A string representing the question we should ask the user next. If blank, the assessment has been completed.

`next_question`: An integer representing the next question that should be asked. You can pass this directly to the next request's `question_number`

`chat_history`: A list of strings representing the updated chat history up until this point. You can pass this directly to the `chat_history` of the next request to maintain the chat history.

for example,
```json
{
  "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
  "next_question": 2,
  "chat_history": [
      "User to System: Hello!",
      "System to User: How confident are you in discussing your maternal health concerns with your healthcare provider?",
  ]
}
```

### Surveys
Surveys are handled through the `/v1/survey` endpoint. It receives a POST request with a JSON body, with the following fields:

`survey_id`: Which survey this is for. Currently there is one survey, `anc`

`user_input`: The message that the user sent to us, as a string

`user_context`: A dictionary of contact fields that we want to send to the LLM

`chat_history`: (optional) a list of strings of the chat history up until this point (not including `user_input`)

for example:
```json
{
  "survey_id": "anc",
  "user_input": "I am 20 weeks pregnant",
  "user_context": {},
  "chat_history": []
}
```

The API will respond with the following fields:

`question`: The next question to ask the user, as a string. If the survey is complete, this may contain a final thank you message.

`user_context`: An updated dictionary of contact fields based on the user's answers.

`chat_history`: A list of strings of the updated chat history. You can pass this directly to the next request to maintain chat history.

`survey_complete`: A boolean flag indicating if the survey has been completed.

for example:
```json
{
  "question": "Have you had any health problems during this pregnancy?",
  "user_context": {
    "weeks_pregnant": 20
  },
  "chat_history": [
    "User to System: I am 20 weeks pregnant",
    "System to User: Have you had any health problems during this pregnancy?"
  ],
  "survey_complete": false
}
```

## Linting
Run the following commands from the root directory:
- `uv run ruff check --fix`
- `uv run ruff format`
- `uv run mypy .`

## Testing
There are automated tests, they can be run with:
- `uv run pytest`
