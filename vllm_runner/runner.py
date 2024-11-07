

llm = LLM(
            model=save_dir,
            tokenizer=model_name,
            dtype='bfloat16',
            distributed_executor_backend="mp",
            tensor_parallel_size=num_gpus_vllm,
            gpu_memory_utilization=gpu_utilization_vllm,
            enable_lora=False,
        )

sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=max_new_tokens,
        stop=stop_tokens
    )

completions = llm.generate(
            prompts,
            sampling_params,
        )