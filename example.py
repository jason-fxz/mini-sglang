from mini_sglang import Engine, ServerArgs

if __name__ == "__main__":
    # Example usage of the Engine class
    server_args = ServerArgs(
        model="/home/jasonfan/huggingface/Qwen3-0.6B",
        attention_backend="fa3",
        gpu_memory_utilization=0.7,
        log_level="DEBUG",  # Set the logging level
    )
    engine = Engine(server_args)

    # Example generate call
    result = engine.generate(
        prompt="All prime numbers within 100 are: 2, 3, ",
        sampling_params={
            "temperature": 0.3,
            "max_new_tokens": 1024,
        },
    )
    print(result)

    result = engine.generate(prompt="The capital of China is", stream=True)
    for chunk in result:
        print(chunk)

    print("Generation completed.")
