from mini_sglang import Engine, ServerArgs
from types import GeneratorType


def print_result(result):
    if isinstance(result, GeneratorType):
        for item in result:
            print(item)
    else:
        print(result)

if __name__ == "__main__":
    # Example usage of the Engine class
    server_args = ServerArgs(
        model="/disk1/models/Qwen3-8B",
        attention_backend="fa3",
        gpu_memory_utilization=0.4,
        tp_size=2,
        log_level="DEBUG",  # Set the logging level
    )
    engine = Engine(server_args)

    # Example generate call
    result = engine.generate(
        prompt="All prime numbers within 100 are: 2, 3, ",
        sampling_params={
            "temperature": 0.0,
            "max_new_tokens": 20,
        },
        stream=False,
    )
    print_result(result)


    result = engine.generate(prompt="The capital of China is", stream=False)
    print_result(result)

    print("Generation completed.")
