# survey_orchestrator.py
import logging
import uuid

from haystack.dataclasses import ChatMessage

# --- Reuse your existing project modules ---
from . import crud, tasks, doc_store
from .enums import HistoryType, Intent, TurnState, ReminderType
from .pydantic_models import (
    LegacySurveyResponse as SurveyResponse,
    OrchestratorSurveyRequest as SurveyRequest,
    OrchestratorUserJourneyState as UserJourneyState,
    SurveyTurnContext,
    ReengagementInfo,
)

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
MAX_REPAIR_STRIKES = 2


# --- Main Orchestrator ---
async def process_survey_turn(request: SurveyRequest) -> SurveyResponse:
    context = await _initialize_turn_context(request)
    try:
        return await _handle_turn(context)
    except Exception:
        logger.exception(
            "survey_turn_unhandled_exception", extra={"trace_id": context.trace_id}
        )
        return SurveyResponse(
            question="Sorry, we’ve run into a technical problem. Please try again later.",
            user_context=request.user_context or {},
            survey_complete=True,
            intent=Intent.SYSTEM_ERROR.value,
            intent_related_response=None,
            results_to_save=[],
            failure_count=request.failure_count,
        )


# --- State Machine & Routing ---
async def _initialize_turn_context(request: SurveyRequest) -> SurveyTurnContext:
    """Creates the fully hydrated SurveyTurnContext object by loading history and journey state."""
    journey_alchemy = await crud.get_user_journey_state(request.user_id)
    history_messages = await crud.get_or_create_chat_history(
        request.user_id, HistoryType(request.survey_id)
    )

    # FIX: Add the user's current input to the history right at the start of the turn.
    if request.user_input:
        history_messages.append(ChatMessage.from_user(text=request.user_input))

    journey_pydantic = (
        UserJourneyState.model_validate(journey_alchemy.__dict__)
        if journey_alchemy
        else None
    )

    previous_context = (
        journey_pydantic.user_context.copy()
        if journey_pydantic
        else request.user_context.copy()
    )
    # The last assistant message is now the second to last message in the list
    last_assistant_message = next(
        (
            msg
            for msg in reversed(history_messages[:-1])
            if msg.role.value == "assistant"
        ),
        None,
    )

    return SurveyTurnContext(
        request=request,
        journey_state=journey_pydantic,
        history=history_messages,
        previous_context=previous_context,
        current_context=previous_context.copy(),
        last_assistant_message=last_assistant_message,
        turn_id=request.turn_id or str(uuid.uuid4()),
        trace_id=request.trace_id or str(uuid.uuid4()),
    )


def _compute_turn_state(ctx: SurveyTurnContext) -> TurnState:
    """Computes the single, definitive state for the current turn."""
    step_id = ctx.journey_state.current_step_identifier if ctx.journey_state else None

    if ctx.request.is_re_engagement_ping:
        return TurnState.RE_ENGAGEMENT

    if not ctx.request.user_input and not ctx.journey_state:
        return TurnState.NEW_SURVEY

    # This is now the single, simple check for any intro response.
    if step_id == "intro":
        return TurnState.AWAITING_INTRO_REPLY

    return TurnState.ACTIVE_TURN


async def _handle_turn(ctx: SurveyTurnContext) -> SurveyResponse:
    state = _compute_turn_state(ctx)
    router = {
        TurnState.RE_ENGAGEMENT: _handle_re_engagement,
        TurnState.NEW_SURVEY: _handle_new_survey,
        TurnState.AWAITING_INTRO_REPLY: _handle_intro_response,
        TurnState.ACTIVE_TURN: _process_active_response,
    }
    return await router[state](ctx)


# --- State Handlers ---
async def _handle_re_engagement(ctx: SurveyTurnContext) -> SurveyResponse:
    if not ctx.journey_state or not ctx.journey_state.last_question_sent:
        return await _handle_new_survey(ctx)
    return await _finalise_and_respond(
        ctx,
        assistant_message=f"Welcome back! Let's continue.\n\n{ctx.journey_state.last_question_sent}",
        next_step_id=ctx.journey_state.current_step_identifier,
    )


async def _handle_new_survey(ctx: SurveyTurnContext) -> SurveyResponse:
    """
    Handles the first turn of a survey by sending the correct static intro message.
    """
    # This logic is now simpler and correctly uses your static messages.
    if ctx.request.survey_id == "anc-survey":
        # Use the specific intro message for the ANC survey
        intro_message = doc_store.INTRO_MESSAGES.get("survey_intro")
    else:
        # Use the default intro for other flows like onboarding
        intro_message = doc_store.INTRO_MESSAGES.get("free_text_intro")

    if not intro_message:
        # Fallback in case the intro message is not found
        logger.error(
            f"Could not find intro message for survey '{ctx.request.survey_id}'"
        )
        intro_message = "Welcome! Shall we begin?"

    # The next step for ANY new survey is now consistently "intro"
    return await _finalise_and_respond(
        ctx, assistant_message=intro_message, next_step_id="intro"
    )


async def _handle_intro_response(ctx: SurveyTurnContext) -> SurveyResponse:
    """
    Handles the user's reply to ANY intro message by dispatching to the
    correct reliable classifier based on the survey_id.
    """
    logger.info(
        f"[{ctx.turn_id}] Handling intro response for survey '{ctx.request.survey_id}'."
    )
    user_input = ctx.request.user_input or ""

    # --- Dispatcher based on survey_id ---
    if ctx.request.survey_id == "anc-survey":
        # For the ANC survey, use the dedicated 3-option classifier
        classification = tasks.classify_anc_start_response(user_input)

        if classification:  # Returns 'YES', 'NO', or 'SOON'
            logger.info(f"[{ctx.turn_id}] ANC intro classified as: {classification}")
            ctx.current_context["intro"] = classification  # Store result under 'intro'
            return await _conclude_valid_turn(ctx)
        else:  # Ambiguous
            logger.warning(
                f"[{ctx.turn_id}] ANC intro response is ambiguous. Triggering repair."
            )
            repair_message = (
                "Sorry, I didn't quite get that. Please reply with a, b, or c."
            )
            return await _handle_repair(ctx, "intro", repair_message)
    else:
        print(f"[{ctx.turn_id}] Unsupported survey type: {ctx.request.survey_id}")
        repair_message = (
            "Sorry, I didn't quite understand. Could you say that differently?"
        )
        return await _handle_repair(ctx, "intro", repair_message)


async def _process_active_response(ctx: SurveyTurnContext) -> SurveyResponse:
    """
    Handles a standard active turn with a simplified and corrected logic flow.
    """
    logger.info(
        f"[{ctx.turn_id}] Processing active response: '{ctx.request.user_input}'."
    )

    user_input = ctx.request.user_input or ""
    step_title = ctx.journey_state.current_step_identifier if ctx.journey_state else ""
    last_question = ctx.journey_state.last_question_sent if ctx.journey_state else ""

    # Pass the main context directly to the reliable extraction task
    tasks.extract_anc_data_from_response(
        user_response=user_input,
        user_context=ctx.current_context,  # The task modifies this dict in-place
        step_title=step_title,
        contextualized_question=(last_question or ""),
    )

    # Check if the task successfully updated the context.
    if ctx.current_context != ctx.previous_context:
        logger.info(f"[{ctx.turn_id}] Extraction successful. Context was updated.")
        return await _conclude_valid_turn(ctx)

    # --- Fallback logic if context was NOT updated ---
    logger.warning(
        f"[{ctx.turn_id}] Extraction failed. Falling back to intent analysis."
    )
    intent, _ = tasks.handle_user_message(
        previous_question=(last_question or ""), user_message=user_input
    )
    logger.info(f"[{ctx.turn_id}] Fallback intent detected: {intent}")

    if intent == Intent.REQUEST_TO_BE_REMINDED.value:
        return await _handle_reminder_request(ctx)
    if intent == Intent.SKIP_QUESTION.value:
        if step_title:
            ctx.current_context[step_title] = "Skip"
        return await _conclude_valid_turn(ctx)

    return await _handle_repair_or_system_skip(ctx)


async def _conclude_valid_turn(ctx: SurveyTurnContext) -> SurveyResponse:
    next_q = await tasks.get_anc_survey_question(
        user_id=ctx.request.user_id,
        user_context=ctx.current_context,
        chat_history=ctx.history,
    )
    if not next_q:
        return await _finalise_and_respond(
            ctx,
            assistant_message="Thank you!",
            next_step_id="end_of_survey",
            survey_complete=True,
        )

    return await _finalise_and_respond(
        ctx,
        assistant_message=next_q.get("contextualized_question", "Thank you!"),
        next_step_id=next_q.get("question_identifier", "end_of_survey"),
        survey_complete=bool(next_q.get("is_final_step", False)),
        reengagement_info=next_q.get("reengagement_info"),
    )


async def _finalise_and_respond(
    ctx: SurveyTurnContext,
    *,
    assistant_message: str,
    next_step_id: str,
    survey_complete: bool = False,
    reengagement_info: ReengagementInfo | None = None,
) -> SurveyResponse:
    diff = {
        k: v for k, v in ctx.current_context.items() if v != ctx.previous_context.get(k)
    }

    # FIX: The user's message is already in the history. We only need to append the assistant's reply.
    ctx.history.append(
        ChatMessage.from_assistant(
            text=assistant_message, meta={"step_id": next_step_id}
        )
    )

    await crud.save_user_journey_state(
        user_id=ctx.request.user_id,
        flow_id=ctx.request.survey_id,
        step_identifier=next_step_id,
        last_question=assistant_message,
        user_context=ctx.current_context,
    )
    await crud.save_chat_history(
        ctx.request.user_id, ctx.history, HistoryType(ctx.request.survey_id)
    )

    return SurveyResponse(
        question=assistant_message,
        question_identifier=next_step_id,
        user_context=ctx.current_context,
        survey_complete=survey_complete,
        intent=Intent.JOURNEY_RESPONSE.value,
        intent_related_response=None,
        results_to_save=sorted(list(diff.keys())),
        failure_count=0,
        reengagement_info=reengagement_info,
    )


async def _handle_repair(
    ctx: SurveyTurnContext, step: str, message: str | None
) -> SurveyResponse:
    repair_message = (
        message or "Sorry, I didn't quite understand. Could you say that differently?"
    )
    return await _finalise_and_respond(
        ctx, assistant_message=repair_message, next_step_id=step
    )


async def _handle_repair_or_system_skip(ctx: SurveyTurnContext) -> SurveyResponse:
    step = (
        ctx.journey_state.current_step_identifier
        if ctx.journey_state
        else "unknown_step"
    )

    if ctx.request.failure_count + 1 >= MAX_REPAIR_STRIKES:
        if step != "unknown_step":
            ctx.current_context[step] = "Skipped - System"
        return await _conclude_valid_turn(ctx)

    # FIX: Define and provide default "" values for optional fields
    last_question = ctx.journey_state.last_question_sent if ctx.journey_state else ""
    user_input = ctx.request.user_input or ""

    repair_message = tasks.handle_conversational_repair(
        flow_id=ctx.request.survey_id,
        question_identifier=(step or 0),
        previous_question=(last_question or ""),
        invalid_input=user_input,
    )

    return SurveyResponse(
        question=repair_message or "Please try again.",
        question_identifier=step,
        user_context=ctx.previous_context,
        survey_complete=False,
        intent="REPAIR",
        intent_related_response=None,
        results_to_save=[],
        failure_count=ctx.request.failure_count + 1,
    )


async def _handle_reminder_request(ctx: SurveyTurnContext) -> SurveyResponse:
    step = ctx.journey_state.current_step_identifier if ctx.journey_state else "intro"
    # FIX: Provide a default "" to satisfy the function's type hint
    last_question = ctx.journey_state.last_question_sent if ctx.journey_state else ""

    message, reengagement_info = await tasks.handle_reminder_request(
        user_id=ctx.request.user_id,
        flow_id=ctx.request.survey_id,
        step_identifier=step,
        last_question=(last_question or ""),
        user_context=ctx.current_context,
        reminder_type=ReminderType.USER_REQUESTED,
    )
    return await _finalise_and_respond(
        ctx,
        assistant_message=message,
        next_step_id=step,
        reengagement_info=reengagement_info,
    )
