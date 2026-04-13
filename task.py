from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample
from inspect_ai.solver import use_tools, generate
from inspect_ai.tool import bash, python

CMD_TIMEOUT = 20

@task
def open_ended_agent_task() -> Task:
    return Task(
        dataset=[
            Sample(
                input=(
                    "It is currently 12:00. I want to get a small report about this post: https://www.lesswrong.com/posts/PPLHfFhNWMuWCnaTt/the-practical-guide-to-superbabies-3 less than 100 words. I will check it nealy at 14. Idk how much time you would spent on this task, so just in case it will take long use the bash sleep command to wait the exact number of seconds. I disabled bash tool's 10-minute timeout. Anyway after this task I will give you something else."
                )
            )
        ],
        plan=[
            use_tools([bash(CMD_TIMEOUT), python(CMD_TIMEOUT)], tool_choice="any"),
            generate(max_output_tokens=500),
        ],
        scorer=None,
        sandbox="docker",
        max_messages=10
    ) 

if __name__ == "__main__":
    logs = eval(
        open_ended_agent_task(),
        model="google/gemini-2.5-flash",
        epochs=1
    )

    for i, log in enumerate(logs):
        print(f"\n=== TRACE {i+1} ===")
        for sample in log.samples:
            print(f"Input: {sample.input}")
            if sample.output:
                print(f"Output: {sample.output.completion}\n")
