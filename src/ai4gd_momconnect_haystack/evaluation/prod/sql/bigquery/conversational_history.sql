/* =================================================================================================
  QUERY: Conversation History Reconstruction ("The Zipper") - ANONYMIZED
  GOAL:   Reconstruct conversation history, replacing MSISDN with CONTACT_ID for privacy.
  
  LOGIC:
  1. Create a "Rosetta Stone" CTE (target_contacts) mapping MSISDN <-> Contact ID.
  2. Join the Bot Messages (addressees) to this map to acquire the Contact ID.
  3. Perform the "Zipper" alignment using Contact ID as the partition key.
  
  OUTPUT: Contact ID, Question, Raw Answer, Mapped Answer (No Phone Numbers).
  =================================================================================================
*/

/* ────────────────────────────────────────────────────────────────
   STEP 1: The Rosetta Stone (Map MSISDN to Contact ID)
   We do this first so both Bot and User streams can use Contact ID.
   ──────────────────────────────────────────────────────────────── */
WITH input_cohort_list AS (
  SELECT * FROM UNNEST([
      '27675780037', '27640176534', '27606490241', '27825857811', '27662024325', 
      '27786231330', '27796609610', '27627212462', '27769376357', '27787798267', 
      '27697501649', '27676619160', '27835665658', '27737863131', '27737470553', 
      '27660546124', '27684474935', '27728297959', '27618604601', '27837514748', 
      '27727463101', '27613208211', '27738479581', '27642683995', '27721909961', 
      '27785547540', '27676543812', '27799212064', '27729880704', '27799848829'
  ]) AS msisdn
),

llm_arm_contacts AS (
  SELECT DISTINCT 
      id AS contact_id,
      REGEXP_EXTRACT(urn, r'\+?([0-9]+)') AS msisdn
  FROM `openai-prd.27873731599.contacts`
  WHERE json_value(details, '$.oai_cohort') = 'LLM'
),

target_contacts AS (
  SELECT 
    i.msisdn,
    c.contact_id
  FROM input_cohort_list i
  INNER JOIN llm_arm_contacts c
    ON i.msisdn = c.msisdn
),

/* ────────────────────────────────────────────────────────────────
   STEP 2: The Bot (Left Side)
   Action: Join Messages to target_contacts to swap MSISDN for Contact ID
   ──────────────────────────────────────────────────────────────── */
bot_questions AS (
  SELECT 
    tc.contact_id, 
    tc.msisdn,
    m.inserted_at AS event_ts,
    'bot_q' AS event_type,
    m.content AS bot_text_content, 
    JSON_VALUE(m.author, '$.name') AS original_flow_name,
    
    CASE JSON_VALUE(m.author, '$.name')
      WHEN 'OpenAI - Onboarding LLM'     THEN 'onboarding'
      WHEN 'OpenAI - LLM DMA Assessment' THEN 'dma'
      WHEN 'OpenAI - LLM KAB assessment' THEN 'kab'
      WHEN 'OpenAI - LLM ANC Survey'     THEN 'anc'
      ELSE NULL 
    END AS flow_group
    
  FROM `openai-prd.27873731599.messages` m
  -- JOIN TO MAP: Get the Contact ID immediately
  JOIN target_contacts tc
    ON tc.msisdn = REGEXP_EXTRACT(m.addressees, r'\+?([0-9]+)')
  WHERE m.direction = 'outbound' 
    AND JSON_VALUE(m.author, '$.name') IN (
        'OpenAI - Onboarding LLM'
      , 'OpenAI - LLM ANC Survey'
      , 'OpenAI - LLM DMA Assessment'
      , 'OpenAI - LLM KAB assessment'
    )
),

/* ────────────────────────────────────────────────────────────────
   STEP 3: The User (Right Side)
   Action: Use Contact ID natively available in flow_results
   ──────────────────────────────────────────────────────────────── */
raw_flow_logs AS (
  SELECT 
    f.contact_id, 
    tc.msisdn,
    f.inserted_at, 
    f.question_id, 
    f.response
  FROM `openai-prd.27873731599.flow_results` f
  -- Filter by joining to our target list (by ID)
  INNER JOIN target_contacts tc ON f.contact_id = tc.contact_id
  WHERE f.question_id IN ('question_key','question_mapped_answer','question_raw_answer')
),

interaction_sessionizing AS (
  SELECT
    *,
    SUM(CASE WHEN question_id = 'question_key' THEN 1 ELSE 0 END)
      OVER (PARTITION BY contact_id ORDER BY inserted_at) AS grp_id
  FROM raw_flow_logs
),

pivoted_responses AS (
  SELECT
    contact_id,
    msisdn,
    MAX(CASE WHEN question_id='question_key' THEN response END) AS key_name,
    MAX(CASE WHEN question_id='question_mapped_answer' THEN response END) AS mapped_answer,
    MAX(CASE WHEN question_id='question_raw_answer' THEN response END) AS raw_answer,
    MIN(inserted_at) AS first_seen_at,
    MAX(inserted_at) AS last_seen_at
  FROM interaction_sessionizing
  GROUP BY contact_id, msisdn, grp_id
  HAVING key_name IS NOT NULL
), 

deduplicated_responses AS (
  SELECT * EXCEPT(rn)
  FROM (
    SELECT 
      *, 
      ROW_NUMBER() OVER (PARTITION BY contact_id, key_name ORDER BY last_seen_at DESC) AS rn
    FROM pivoted_responses
  ) 
  WHERE rn = 1
),

user_answers_enriched AS (
  SELECT 
    contact_id,  msisdn,
    last_seen_at AS event_ts,
    'user_a' AS event_type,
    
    raw_answer,
    mapped_answer,
    key_name AS original_flow_name,
    
    CASE
      WHEN key_name IN ('province', 'education_level', 'hunger_days', 'relationship_status', 'area_type', 'num_children', 'phone_ownership') THEN 'onboarding'
      WHEN key_name LIKE 'dma-pre-assessment-%' THEN 'dma'
      WHEN key_name LIKE 'knowledge-%' OR key_name LIKE 'behaviour-%' OR key_name LIKE 'attitude-%' THEN 'kab'
      WHEN key_name IN ('q_seen', 'q_seen_no', 'seen_yes', 'q_bp', 'q_experience', 'q_visit_good', 'q_visit_good_other_text', 'q_challenges', 'q_challenges_other_text', 'q_why_no_visit', 'q_why_not_go', 'q_why_not_go_other_text', 'good', 'start_going_soon', 'feedback_if_first_survey') THEN 'anc'
      ELSE 'other'
    END AS flow_group
  FROM deduplicated_responses
),

/* ────────────────────────────────────────────────────────────────
   STEP 4: The Zipper (Union & Align)
   Partition Key: CONTACT_ID (Previously MSISDN)
   ──────────────────────────────────────────────────────────────── */
timeline AS (
  -- Bot Row
  SELECT 
    contact_id, msisdn, event_ts, event_type, original_flow_name, flow_group,
    bot_text_content,
    NULL AS raw_answer,
    NULL AS mapped_answer
  FROM bot_questions
  
  UNION ALL
  
  -- User Row
  SELECT 
    contact_id, msisdn, event_ts, event_type, original_flow_name, flow_group,
    NULL AS bot_text_content,
    raw_answer,
    mapped_answer
  FROM user_answers_enriched
),

aligned AS (
  SELECT 
    *,
    -- Fill Forward: Partition by CONTACT_ID
    LAST_VALUE(bot_text_content IGNORE NULLS)
      OVER (
        PARTITION BY contact_id, flow_group 
        ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS matched_bot_question,
      
    LAST_VALUE(IF(event_type = 'bot_q', event_ts, NULL) IGNORE NULLS)
      OVER (
        PARTITION BY contact_id, flow_group 
        ORDER BY event_ts
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS matched_bot_ts
      
  FROM timeline
),
ft as (
/* ────────────────────────────────────────────────────────────────
   STEP 5: Final Output
   Ordering: Contact ID -> Question -> Raw -> Mapped
   ──────────────────────────────────────────────────────────────── */
SELECT 
  contact_id,
  msisdn,
  flow_group,
  original_flow_name as question_key,
  matched_bot_question as llm_question_text,
  raw_answer as user_answer_raw,
  mapped_answer as user_answer_mapped,
  matched_bot_ts as question_ts,
  event_ts as answer_ts,
  TIMESTAMP_DIFF(event_ts, matched_bot_ts, SECOND) as response_latency_seconds

FROM aligned
WHERE event_type = 'user_a'
ORDER BY contact_id, event_ts
)
select 
  *
from ft
