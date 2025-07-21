# python 1_generate_meta_domain.py \
#     --model_name_or_path ../models/qwen3-32b \
#     --output_file data/meta_domains.txt

# python 2_generate_meta_request.py \
#     --model_name_or_path ../models/qwen3-32b \
#     --domain_file data/meta_domains.txt \
#     --output_file data/meta_requests.txt

# python 3_generate_meta_scenario.py \
#     --model_name_or_path ../models/qwen3-32b \
#     --input_file data/meta_requests.txt \
#     --output_file data/meta_scenarios.txt

# python 4_generate_meta_persona.py \
#     --model_name_or_path ../models/qwen3-32b \
#     --input_file data/meta_scenarios.txt \
#     --output_file data/meta_personas.txt

# python 5_generate_primary_type.py \
#     --model ../models/qwen3-32b \
#     --output data/primary_type.txt


# python 6_generate_llm_based_constraint.py \
#     --model ../models/qwen3-32b \
#     --primary data/primary_type.txt \
#     --output data/llm_based_constraints.json 

python 7_generate_instruction.py

python 8_generate_response.py \
    --model_name_or_path ../models/qwen3-8b \
    --input_json data/instructions.json \
    --output_json data/responses.json

python 9_reject_sampling_sft.py

# python process_rl_data.py

# cd verl
# bash run_grpo.sh
