import asyncio
import logging
import time
import uuid
from typing import Any
from dataclasses import dataclass, field

from haystack.dataclasses import ChatMessage

# --- Reuse your existing project modules ---
from . import crud, tasks, doc_store
from .enums import HistoryType, Intent, TurnState, ReminderType
from .pydantic_models import (
    SurveyRequest,
    SurveyResponse,
    UserJourneyState as PydanticUserJourneyState, # Use an alias to avoid name clash
    ReengagementInfo,
)

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
MAX_REPAIR_STRIKES = 2

# --- Dataclass for internal state management ---
@dataclass
class SurveyTurnContext:
    request: SurveyRequest
    journey_state: PydanticUserJourneyState | None = None
    history: list[ChatMessage] = field(default_factory=list)
    previous_context: dict[str, Any] = field(default_factory=dict)
    current_context: dict[str, Any] = field(default_factory=dict)
    last_assistant_message: ChatMessage | None = None
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))

# --- Main Orchestrator ---
async def process_survey_turn(request: SurveyRequest) -> SurveyResponse:
    context = await _initialize_turn_context(request)
    try:
        return await _handle_turn(context)
    except Exception:
        logger.exception("survey_turn_unhandled_exception", extra={"trace_id": context.trace_id})
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
    journey_alchemy = await crud.get_user_journey_state(request.user_id)
    history_messages = await crud.get_or_create_chat_history(request.user_id, HistoryType(request.survey_id))
    
    journey_pydantic = PydanticUserJourneyState.model_validate(journey_alchemy.__dict__) if journey_alchemy else None
    
    previous_context = journey_pydantic.user_context.copy() if journey_pydantic else request.user_context.copy()
    last_assistant_message = next((msg for msg in reversed(history_messages) if msg.role.value == "assistant"), None)
    
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
    # FIX: Use `current_step_identifier` from the original SQLAlchemy model
    step_id = ctx.journey_state.current_step_identifier if ctx.journey_state else None
    if ctx.request.is_re_engagement_ping:
        return TurnState.RE_ENGAGEMENT
    if not ctx.request.user_input and not ctx.journey_state:
        return TurnState.NEW_SURVEY
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
    intro_message = doc_store.INTRO_MESSAGES["free_text_intro"]
    return await _finalise_and_respond(ctx, assistant_message=intro_message, next_step_id="intro")

async def _handle_intro_response(ctx: SurveyTurnContext) -> SurveyResponse:
    # FIX: Ensure user_input is not None before passing
    user_input = ctx.request.user_input or ""
    result = tasks.handle_intro_response(user_input=user_input, flow_id=ctx.request.survey_id)
    
    if result.get("action") == "PROCEED":
        return await _conclude_valid_turn(ctx)
    elif result.get("action") == "PAUSE_AND_REMIND":
        return await _handle_reminder_request(ctx)
    else:
        return await _handle_repair(ctx, "intro", result.get("message"))

async def _process_active_response(ctx: SurveyTurnContext) -> SurveyResponse:
    updated_context = ctx.previous_context.copy()
    
    # REUSE: Your existing extraction task, which returns the full updated context
    updated_context, _ = tasks.extract_anc_data_from_response(
        user_response=ctx.request.user_input or "",
        user_context=updated_context,
        step_title=ctx.journey_state.current_step_identifier if ctx.journey_state else "",
        contextualized_question=ctx.journey_state.last_question_sent if ctx.journey_state else "",
    )
    
    if updated_context != ctx.previous_context:
        ctx.current_context = updated_context
        return await _conclude_valid_turn(ctx)

    # Fallback to intent analysis if extraction fails
    intent, _ = tasks.handle_user_message(
        previous_question=ctx.journey_state.last_question_sent if ctx.journey_state else "",
        user_message=ctx.request.user_input or "",
    )

    if intent == Intent.REQUEST_TO_BE_REMINDED.value:
        return await _handle_reminder_request(ctx)
    if intent == Intent.SKIP_QUESTION.value:
        step = ctx.journey_state.current_step_identifier if ctx.journey_state else "unknown_step"
        ctx.current_context[step] = "Skip"
        return await _conclude_valid_turn(ctx)
        
    return await _handle_repair_or_system_skip(ctx)

# --- Conclusion, Persistence & Repair ---
async def _conclude_valid_turn(ctx: SurveyTurnContext) -> SurveyResponse:
    next_q = await tasks.get_anc_survey_question(
        user_id=ctx.request.user_id, user_context=ctx.current_context, chat_history=ctx.history
    )
    if not next_q:
        return await _finalise_and_respond(ctx, assistant_message="Thank you!", next_step_id="end_of_survey", survey_complete=True)
        
    return await _finalise_and_respond(
        ctx,
        assistant_message=next_q.get("contextualized_question", "Thank you!"),
        next_step_id=next_q.get("question_identifier", "end_of_survey"),
        survey_complete=bool(next_q.get("is_final_step", False)),
        reengagement_info=next_q.get("reengagement_info")
    )

async def _finalise_and_respond(
    ctx: SurveyTurnContext, *, assistant_message: str, next_step_id: str, survey_complete: bool = False, reengagement_info: ReengagementInfo | None = None
) -> SurveyResponse:
    diff = {k: v for k, v in ctx.current_context.items() if v != ctx.previous_context.get(k)}
    
    if ctx.request.user_input:
        ctx.history.append(ChatMessage.from_user(text=ctx.request.user_input))
    ctx.history.append(ChatMessage.from_assistant(text=assistant_message, meta={"step_id": next_step_id}))

    # FIX: Call save_user_journey_state with its original signature
    await crud.save_user_journey_state(
        user_id=ctx.request.user_id,
        flow_id=ctx.request.survey_id,
        step_identifier=next_step_id,
        last_question=assistant_message,
        user_context=ctx.current_context,
    )
    await crud.save_chat_history(ctx.request.user_id, ctx.history, HistoryType(ctx.request.survey_id))

    # FIX: Construct the original SurveyResponse model with all its required fields
    return SurveyResponse(
        question=assistant_message,
        question_identifier=next_step_id,
        user_context=ctx.current_context,
        survey_complete=survey_complete,
        intent=Intent.JOURNEY_RESPONSE.value,
        intent_related_response=None,
        results_to_save=sorted(list(diff.keys())),
        failure_count=0, # Reset failure count on a successful turn
        reengagement_info=reengagement_info,
    )

async def _handle_repair(ctx: SurveyTurnContext, step: str, message: str | None) -> SurveyResponse:
    repair_message = message or "Sorry, I didn't quite understand."
    return await _finalise_and_respond(ctx, assistant_message=repair_message, next_step_id=step)

async def _handle_repair_or_system_skip(ctx: SurveyTurnContext) -> SurveyResponse:
    step = ctx.journey_state.current_step_identifier if ctx.journey_state else "unknown_step"
    
    if ctx.request.failure_count + 1 >= MAX_REPAIR_STRIKES:
        if step: ctx.current_context[step] = "Skipped - System"
        return await _conclude_valid_turn(ctx)
    
    repair_message = tasks.handle_conversational_repair(
        flow_id=ctx.request.survey_id,
        question_identifier=step or 0,
        previous_question=ctx.journey_state.last_question_sent if ctx.journey_state else "",
        invalid_input=ctx.request.user_input or "",
    )
    # FIX: Return a SurveyResponse that signals repair
    return SurveyResponse(
        question=repair_message or "Please try again.",
        question_identifier=step,
        user_context=ctx.previous_context, # Return original context on failure
        survey_complete=False,
        intent="REPAIR",
        intent_related_response=None,
        results_to_save=[],
        failure_count=ctx.request.failure_count + 1,
    )

async def _handle_reminder_request(ctx: SurveyTurnContext) -> SurveyResponse:
    step = ctx.journey_state.current_step_identifier if ctx.journey_state else "intro"
    last_question = ctx.journey_state.last_question_sent if ctx.journey_state else ""
    
    message, reengagement_info = await tasks.handle_reminder_request(
        user_id=ctx.request.user_id,
        flow_id=ctx.request.survey_id,
        step_identifier=step,
        last_question=last_question,
        user_context=ctx.current_context,
        reminder_type=ReminderType.USER_REQUESTED,
    )
    return await _finalise_and_respond(ctx, assistant_message=message, next_step_id=step, reengagement_info=reengagement_info)