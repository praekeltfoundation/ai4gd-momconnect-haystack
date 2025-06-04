# Inside or after run_evaluation_simulation(dataset_item)

# ... simulation runs ...
# Assume `trace_id` is returned by run_evaluation_simulation
# Assume `actual_llm_outputs_onboarding` and `actual_llm_outputs_assessment` are collected during the simulation

# Example for onboarding extraction scoring:
# You'd need to iterate through turns or score at the end of the simulation
# For simplicity, let's assume 'final_extracted_context' holds all data extracted during onboarding
# and 'ground_truth_data_onboarding' is dataset_item['expected_output'] for onboarding
# This requires careful state management in your run_evaluation_simulation to get all actual outputs.

# If scoring after the full simulation:
# 1. Fetch the trace and its observations from Langfuse (if not available locally)
# 2. Identify relevant spans (e.g., data extraction spans)
# 3. Apply scorers

# OR, a simpler approach: Score directly within the simulation if possible,
# or pass necessary data to a post-processing scoring function.

# Let's assume you have `final_user_context` from onboarding
# and `dataset_item['expected_output']` is the ground truth.
# This is highly dependent on how you structure `run_evaluation_simulation` to provide these.

# If your `run_evaluation_simulation` is part of a Langfuse Job:
# Langfuse jobs handle passing trace/observation outputs to scorers.
# You define scorers that receive `output` from observed functions.

# If running manually and scoring after:
# trace_obj = langfuse.get_trace(trace_id)
# For each relevant observation/span in trace_obj.observations:
#   if observation.name == "extract_onboarding_data_span_name": # Needs careful naming of spans
#     llm_extracted_this_turn = observation.output 
#     # ... map to ground truth for this turn ...
#     # extraction_scores = score_extraction_accuracy(trace_id, observation.id, llm_extracted_this_turn, ground_truth_for_this_turn)
#     # for score_obj in extraction_scores: langfuse.score(score_obj)

# This part requires a robust way to align LLM outputs from the trace with ground truth.
# Your `golden_dataset` structure with `simulated_user_responses_map` and `ground_truth_extracted_data`
# is excellent because it directly maps responses to expected extractions.
# Your `run_evaluation_simulation` should be designed to:
#   1. Use `simulated_user_responses_map` to feed user inputs.
#   2. Collect the actual LLM extractions after each relevant turn.
#   3. Then, after the simulation (or per turn), compare with `ground_truth_extracted_data`.

# For MBEs on question contextualization (e.g., in onboarding):
# You'd iterate through the turns where questions were generated.
# For each `get_next_onboarding_question` span:
#   actual_contextualized_q = span.output 
#   user_context_at_that_time = span.input['user_context']
#   chat_history_at_that_time = span.input['chat_history']
#   standard_q_content = ... (from onboarding_flow.json, requires mapping)
#   score_question_quality_mbe(trace_id, span.id, user_context_at_that_time, ...)