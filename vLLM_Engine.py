from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class vLLM_Engine:
    def __init__(self, model_name_or_path, decoding_dict, tp=4):
        self.model_name_or_path = model_name_or_path
        self.decoding_dict = decoding_dict
        self.tp = tp
    
    def _build_engine(self):
        sampling_params = SamplingParams(
            **self.decoding_dict
        )
        vllm_engine = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tp,
            dtype="bfloat16",
        )

        return vllm_engine, sampling_params
    
    def _get_formatted_input(self, prompts):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        formatted_prompts = []
        
        for prompt in prompts:
            messages = []
            
            if "system" in prompt:
                messages.append({"role": "system", "content": prompt["system"]})
            
            if "conversation" in prompt:
                for turn in prompt["conversation"]:
                    role = turn["role"]
                    content = turn["content"]
                    messages.append({"role": role, "content": content})
            
            if "user" in prompt:
                messages.append({"role": "user", "content": prompt["user"]})
            
            chat_message = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            formatted_prompts.append(chat_message)
        
        return formatted_prompts
    
    def _process_outputs(self, outputs):
        responses = []
        for output in outputs:
            response = output.outputs[0].text.strip()
            if '</think>' in response:
                response = response.split("</think>")[1].strip()
            responses.append(response)
        
        return responses
    
    def generate(self, prompts):
        vllm_engine, sampling_params = self._build_engine()
        formatted_prompts = self._get_formatted_input(prompts)
        outputs = vllm_engine.generate(formatted_prompts, sampling_params)
        responses = self._process_outputs(outputs)

        return responses
