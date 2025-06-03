from unittest import mock
from ai4gd_momconnect_haystack.tasks import get_assessment_question


@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_assessment_question(pipelines_mock):
    pipelines_mock.run_assessment_contextualization_pipeline.return_value = "mock_question"
    result = get_assessment_question(
        flow_id="assessment_flow_id",
        question_number=0,
        current_assessment_step=0,
        user_context={}
    )
    assert result == {"contextualized_question": "mock_question", "current_question_number": 1}

@mock.patch("ai4gd_momconnect_haystack.tasks.pipelines")
def test_get_last_assessment_question(pipelines_mock):
    pipelines_mock.run_assessment_contextualization_pipeline.return_value = "mock_question"
    result = get_assessment_question(
        flow_id="assessment_flow_id",
        question_number=4,
        current_assessment_step=4,
        user_context={}
    )
    assert result == {"contextualized_question": "mock_question", "current_question_number": 5}
    result = get_assessment_question(
        flow_id="assessment_flow_id",
        question_number=5,
        current_assessment_step=5,
        user_context={}
    )
    assert result == {}
