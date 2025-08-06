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
    anc = "anc"
    onboarding = "onboarding"


class ReminderType(IntEnum):
    FIRST = 1
    SECOND = 2
