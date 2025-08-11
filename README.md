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

## Database Migrations
We use [alembic](https://alembic.sqlalchemy.org/en/latest/) to manage DB schema changes.

Database migrations need to be generated after making any changes to the sqlalchemy models.

Migrations are run during the apps startup but can also be manually applied.

To generate:
`uv run alembic revision --autogenerate -m "<description of schema changes>"`

To apply manually:
`uv run alembic upgrade head`

## API

### Documentation
The API is documented in this readme, but there is also automatically generated documentation available at the `/docs` endpoint of the service.

### Authentication
Authentication is handled through the `Authorization` HTTP header, the value of which should be `Token <token>`. The token is configured using the `API_TOKEN` environment variable. There is only one token for the whole service.

## QA Reset Feature
- **How it Works:** An admin must first add a QA tester's user ID to the `QA_USER_IDS` environment variable (`QA_USER_IDS="27821112222,27833334444"`). The authorized tester can then send the !reset command at any time during any conversation flow (onboarding, survey, etc.) to trigger the reset.

- **What it Does:** The command permanently deletes all records for that user from five key tables: `ChatHistory`, `AssessmentHistory`, `AssessmentResultHistory`, `AssessmentEndMessagingHistory`, and `UserJourneyState`.

- **Outcome:** This allows the tester to immediately restart any flow from the beginning as if they were a new user.

### Intents
For every response, we first classify the intent of the user's message. If the intent is to answer the qusetion that we asked them, then we should continue as normal. But if it's not, then we should take a different action depending on the intent. There are `intent` and `intent_related_response` fields added to every response to deal with this. The values for `intent` are:

`JOURNEY_RESPONSE` - This is what we expect, the user is answering the question, and we should proceed as normal. There is no intent related response for this intent.

`QUESTION_ABOUT_STUDY` - If the user asks a question about the study instead of answering the question. The `intent_related_response` here will contain an answer for their question, if it's something that's in our bank of FAQs, otherwise it will be "Sorry, I don't have that information right now". We should show this answer to the user to answer their question.

`HEALTH_QUESTION` - We cannot answer health related questions on this study line due to ethical concerns. `intent_related_response` will contain a message redirecting them to the main MomConnect WhatsApp line, where they can get answers to their health related questions.

`ASKING_TO_STOP_MESSAGES` - The user is asking to stop receiving messages, we should take the appropriate action to stop sending them push messages for the study. There is no related response for this intent.

`ASKING_TO_DELETE_DATA` - The user is asking for us to delete their data, we should take the appropriate action to delete all their data that we've collected on the study line. There is no related response for this intent.

`REPORTING_AIRTIME_NOT_RECEIVED` - The user is saying that they did not receive their airtime, we should alert the helpdesk of this user so that we can figure out why they did not receive their airtime. There is no related response for this intent.

`CHITCHAT` - The user is making a conversational comment. There is response to the user in the `intent_related_response`.


### Onboarding
Onboarding is handled through the `/v1/onboarding` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_id`: A unique identifier for a user, as a string

`user_input`: The message that the user sent to us, as a string

`user_context`: A dictionary of contact fields that we want to send to the LLM

for example:
```json
{
  "user_id": "1234",
  "user_input": "Hello!",
  "user_context": {}
}
```

It will respond with the following fields:

`question`: The next question to ask the user, as a string. If not supplied, onboarding is complete.

`user_context`: An updated dictionary of contact fields.

`intent`: (optional) the user's intent. See the ["Intents"](#intents) section for more info.

`intent_related_response`: (optional) a response related to the user's intent. See the ["Intents"](#intents) section for more info.

for example
```json
{
  "user_context": {"area_type": "City"},
  "question": "Which province are you currently living in? 🏡",
  "intent": "JOURNEY_RESPONSE",
  "intent_related_response": null
}
```

### Assessment
Assessments are handled through the `/v1/assessment` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_id`: A unique identifier for a user, as a string

`user_input`: The user's message, as a string

`user_context`: The user's contact fields that we want to share with the LLM

`flow_id`: Which assessment to run, as a string. The following assessments are available: `dma-pre-assessment`, `dma-post-assessment`, `knowledge-pre-assessment`, `knowledge-post-assessment`, `attitude-pre-assessment`, `attitude-post-assessment`, `behaviour-pre-assessment`, `behaviour-post-assessment`

`question_number`: Which question number we are on, as an integer.

`previous_question`: The previous question (of the same assessment) that was sent to the user, as a string

for example:
```json
{
  "user_id": "1234",
  "user_context": {},
  "user_input": "Hello!",
  "flow_id": "dma-assessment",
  "question_number": 1,
}
```

The API responds with the following fields:

`question`: A string representing the question we should ask the user next. If blank, the assessment has been completed.

`next_question`: An integer representing the next question that should be asked. You can pass this directly to the next request's `question_number`

`intent`: (optional) the user's intent. See the ["Intents"](#intents) section for more info.

`intent_related_response`: (optional) a response related to the user's intent. See the ["Intents"](#intents) section for more info.

for example,
```json
{
  "question": "How confident are you in discussing your maternal health concerns with your healthcare provider?",
  "next_question": 2,
  "intent": "JOURNEY_RESPONSE",
  "intent_related_response": null
}
```

### Surveys
Surveys are handled through the `/v1/survey` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_id`: A unique identifier for a user, as a string

`survey_id`: Which survey this is for. Currently there is one survey, `anc`

`user_input`: The message that the user sent to us, as a string

`user_context`: A dictionary of contact fields that we want to send to the LLM

for example:
```json
{
  "survey_id": "anc",
  "user_input": "I am 20 weeks pregnant",
  "user_context": {},
}
```

The API will respond with the following fields:

`question`: The next question to ask the user, as a string. If the survey is complete, this may contain a final thank you message.

`user_context`: An updated dictionary of contact fields based on the user's answers.

`survey_complete`: A boolean flag indicating if the survey has been completed.

`intent`: (optional) the user's intent. See the ["Intents"](#intents) section for more info.

`intent_related_response`: (optional) a response related to the user's intent. See the ["Intents"](#intents) section for more info.

for example:
```json
{
  "question": "Have you had any health problems during this pregnancy?",
  "user_context": {
    "weeks_pregnant": 20
  },
  "survey_complete": false,
  "intent": "JOURNEY_RESPONSE",
  "intent_related_response": null
}
```

### End of pre-assessment journeys
A short thank you and feedback journey is handled through the `/v1/assessment-end` endpoint. It receives a POST request with a JSON body, with the following fields:

`user_id`: A unique identifier for a user, as a string

`flow_id`: Which assessment this is for. This can be one of `dma-pre-assessment`, `knowledge-pre-assessment`, `behaviour-pre-assessment`, `attitude-pre-assessment`

`user_input`: The message that the user sent to us, as a string

for example:
```json
{
  "user_id": "1234",
  "flow_id": "dma-pre-assessment",
  "user_input": "I am 20 weeks pregnant",
}
```

The API will respond with the following fields:

`message`: The next message to send to the user, as a string. If the journey is complete, this may contain a final thank you message.

`task`: (optional) a task that needs to be performed by Turn. This can be one of:
- `REMIND_ME_LATER`: set a reminder to reengage the user in 24 hours.
- `STORE_FEEDBACK`: save a flow result containing validated feedback shared by the user.

`intent`: (optional) the user's intent. See the ["Intents"](#intents) section for more info.

`intent_related_response`: (optional) a response related to the user's intent. See the ["Intents"](#intents) section for more info.

for example:
```json
{
  "message": "Thank you for answering these questions!",
  "task": "",
  "intent": "JOURNEY_RESPONSE",
  "intent_related_response": null
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

## Evaluation
The project includes a script to evaluate the performance of the LLM pipelines against a predefined ground truth.

### Command-Line Reference

The following flags can be used to control the behavior of the `evaluator.py` script. The `Argument` column shows what kind of value the flag expects.

| Flag               | Argument | Description                                                                                                                                              | Default Value              |
| :----------------- | :------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- |
| `--run-simulation` | (None)   | A boolean flag that, when present, executes a new simulation run based on the ground truth file before evaluation.                                         | `False`                    |
| `--save-report`    | (None)   | A boolean flag that, when present, saves the final evaluation report to a text file. The report is named after the results file (e.g., `results.json` -> `results.report.txt`). | `False`                    |
| `--gt-file`        | `[PATH]` | The path to the JSON file containing the ground truth scenarios.                                                                                         | `data/ground_truth.json`   |
| `--results-file`   | `[PATH]` | The path to a specific simulation results JSON file to evaluate. This is ignored if `--run-simulation` is used.                                           | `None`                     |
| `--openai-key`     | `[KEY]`  | Your OpenAI API key.                                                                                                                                     | `OPENAI_API_KEY` env var   |
| `--deepeval-key`   | `[KEY]`  | Your DeepEval API key for semantic evaluations.                                                                                                          | `DEEPEVAL_API_KEY` env var |
| `--eval-model`     | `[MODEL]`| The name of the GPT model to use for the DeepEval semantic metrics.                                                                                        | `gpt-4o`                   |
| `--threshold`      | `[FLOAT]`| The success threshold (0.0 to 1.0) for the DeepEval semantic metrics.                                                                                      | `0.7`                      |

### Ground Truth Files
The ground truth files define the scenarios for the evaluation. They are located in `src/ai4gd_momconnect_haystack/evaluation/data/` and include:
- `onboarding_gt.json`
- `dma_gt.json`
- `kab_gt.json`
- `anc_gt.json`

### How to Run
There are two ways to run the evaluation script:

**1. Generate a New Simulation and Evaluate It (Recommended)**

This method first runs an automated simulation based on a ground truth file and then immediately evaluates the results. Use the `--run-simulation` flag and specify which ground truth file to use with `--gt-file`.

<details>
<summary>Click to view example command</summary>

```bash
docker-compose run --remove-orphans python-app uv run python src/ai4gd_momconnect_haystack/evaluation/evaluator.py \
  --run-simulation \
  --gt-file src/ai4gd_momconnect_haystack/evaluation/data/kab_gt.json \
  --save-report
```
&lt;</details>


**2. Evaluate an Existing Simulation File**

Alternatively, if you have an existing simulation run output, you can evaluate it directly against a ground truth (gt) file. This is done by passing the --results-file flag, which allows you to re-evaluate a past run without re-running the simulation.

<details>
<summary>Click to view example command</summary>

```bash
docker-compose run --remove-orphans python-app uv run python src/ai4gd_momconnect_haystack/evaluation/evaluator.py \
  --results-file src/ai4gd_momconnect_haystack/evaluation/data/simulation_run_results_250624-1132.json \
  --gt-file src/ai4gd_momconnect_haystack/evaluation/data/kab_gt.json \
  --save-report
```
&lt;</details>

**Report Output**

Using the --save-report flag will generate a text file containing both a detailed turn-by-turn analysis and a final performance summary. The report will be saved in the same directory as the results file with a .report.txt extension.