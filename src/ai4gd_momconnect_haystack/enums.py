from enum import Enum, IntEnum


class AssessmentType(str, Enum):
    dma_pre_assessment = "dma-pre-assessment"
    dma_post_assessment = "dma-post-assessment"
    knowledge_pre_assessment = "knowledge-pre-assessment"
    knowledge_post_assessment = "knowledge-post-assessment"
    attitude_pre_assessment = "attitude-pre-assessment"
    attitude_post_assessment = "attitude-post-assessment"
    behaviour_pre_assessment = "behaviour-pre-assessment"
    behaviour_post_assessment = "behaviour-post-assessment"


class HistoryType(str, Enum):
    anc = "anc-survey"
    onboarding = "onboarding"


class ReminderType(IntEnum):
    USER_REQUESTED = 1
    USER_REQUESTED_SECOND = 2
    SYSTEM_SCHEDULED_THREE_DAY = 3


class TurnState(str, Enum):
    RE_ENGAGEMENT = "RE_ENGAGEMENT"
    NEW_SURVEY = "NEW_SURVEY"
    AWAITING_INTRO_REPLY = "AWAITING_INTRO_REPLY"
    ACTIVE_TURN = "ACTIVE_TURN"


class Intent(str, Enum):
    REQUEST_TO_BE_REMINDED = "REQUEST_TO_BE_REMINDED"
    SKIP_QUESTION = "SKIP_QUESTION"
    JOURNEY_RESPONSE = "JOURNEY_RESPONSE"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CHITCHAT = "CHITCHAT"
    JOURNEY_RESUMED = "JOURNEY_RESUMED"


class ExtractionStatus(Enum):
    SUCCESS = "SUCCESS"
    NO_MATCH = "NO_MATCH"
