# self designed AI-agent using Claude Sonnet

# import API key
from dotenv import load_dotenv # type: ignore
load_dotenv()

import anthropic # type: ignore
import subprocess
import os

client = anthropic.Anthropic()

# keep track of dangerous actions
DANGEROUS = ["write_file", "delete_file", "run_command"]

# store always allowed operations
always_allow = set()

# define tools
tools = [
    {"name": "read_file", "description": "Read file contents",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}}, 
                      "required": ["path"]}},
    {"name": "list_dir", "description": "List files in a directory",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}}, 
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file, overwrites existing content",
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}, 
                                     "content": {"type": "string"}}, 
                      "required": ["path", "content"]}},
    {"name": "delete_file", "description": "Delete an existing file", 
     "input_schema": {"type": "object", 
                      "properties": {"path": {"type": "string"}}, 
                      "required": ["path"]}}, 
    {"name": "run_command", "description": "Execute a terminal command and return output",
     "input_schema": {"type": "object", 
                      "properties": {"command": {"type": "string"}}, 
                      "required": ["command"]}},
    {"name": "search_file", "description": "Search for a pattern inside a file, returns matching lines", 
     "input_schema": {"type": "object",
                  "properties": {"path": {"type": "string"},
                                 "pattern": {"type": "string", "description": "text to search for"}},
                  "required": ["path", "pattern"]}}, 
    {"name": "create_csv", "description": "Generate Latin hypercube samples and save to CSV file", 
     "input_schema": {"type": "object",
                  "properties": {
                      "filename": {"type": "string"},
                      "n_samples": {"type": "integer", "description": "number of rows"},
                      "n_params": {"type": "integer", "description": "number of parameters (columns)"}},
                  "required": ["filename", "n_samples", "n_params"]}}
]

# execute tools
def execute_tool(name, params): 
    # ask for permission before using dangerous tools
    if name in DANGEROUS and name not in always_allow: 
        print(f"\n[Permission] {name} | params: {params}")
        c = input("  Allow? [y=once / a=always this session / n=deny]: ")
        if c == "n":
            return "User denied this operation"
        if c == "a":
            always_allow.add(name)

    if name == "read_file":
        return open(params["path"]).read()
    if name == "list_dir":
        return str(os.listdir(params["path"]))
    if name == "write_file":
        open(params["path"], "w").write(params["content"])
        return f"Written to {params['path']}"
    if name == "delete_file": 
        os.remove(params["path"])
        return f"Deleted {params['path']}"
    if name == "run_command":
        result = subprocess.run(params["command"], shell=True, capture_output=True, text=True)
        return result.stdout or result.stderr
    if name == "search_file":
        pattern = params["pattern"] # word/phrase to look for
        results = []
        with open(params["path"]) as f:
            for i, line in enumerate(f, 1): # i = line number (starts at 1)
                if pattern in line: # if pattern is in this line
                    results.append(f"line {i}: {line.rstrip()}") # rstrip removes trailing newline
        return "\n".join(results) if results else "No matches found"
    if name == "create_csv":
        import csv
        from scipy.stats import qmc # type: ignore
        sampler = qmc.LatinHypercube(d=params["n_params"]) # d = number of parameters
        samples = sampler.random(n=params["n_samples"]) # generate samples
        with open(params["filename"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"param_{i+1}" for i in range(params["n_params"])]) # header row
            writer.writerows(samples) # write all data
        return f"Saved {params['n_samples']} samples to {params['filename']}"

# chat history
messages = []

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ("exit", "quit"):
        break
    messages.append({"role": "user", "content": user_input})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            tools=tools,
            system="""You are an expert coding assistant.

Rules:
- Always read a file before editing it
- Before running any command, explain what it does
- If you're unsure about something, ask before acting
- Keep responses concise and technical

You have tools to read/write files and run terminal commands. 
Use them proactively to give accurate answers instead of guessing.""", # system prompt
            messages=messages
        )

        # terminate only when words are returned
        if response.stop_reason == "end_turn":
            assistant_message = response.content[0].text
            print("Claude:", assistant_message)
            messages.append({"role": "assistant", "content": assistant_message})
            break

        # tool use
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
    
            # collect results for all tool calls in this response
            tool_results = []
            for tc in response.content:
                if tc.type == "tool_use":
                    print(f"[Tool] {tc.name}")
                    result = execute_tool(tc.name, tc.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": result
                    })
    
            # add all results in one message
            messages.append({"role": "user", "content": tool_results})

