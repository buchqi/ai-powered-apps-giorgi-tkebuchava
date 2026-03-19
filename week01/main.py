"""
CS-AI-2025 | Lab 1 | Exercise 1 — Hello AI
Spring 2026 | Kutaisi International University

Your very first call to a language model API.

Before running this script:
1. Complete tools-setup.md
2. Create a .env file with your GEMINI_API_KEY
3. Activate your virtual environment: source venv/bin/activate
4. Run: python examples/starter-code/01_hello_gemini.py
"""

import os
import time
from dotenv import load_dotenv
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Load environment variables from .env file
# This MUST happen before importing google.genai in some configurations
load_dotenv()

try:
    import google.genai as genai
except ImportError:
    print("ERROR: google-genai package not installed.")
    print("Run: pip install google-genai")
    exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────

MODEL = "gemini-3-flash-preview"

PROMPT = (
    "Tell me one fascinating, non-obvious thing about how large language models "
    "work internally — something that would genuinely surprise a computer science student "
    "who has never studied AI. Answer in exactly two sentences."
)

def update_cost_analysis(model_name, input_tokens, output_tokens, total_tokens, latency_ms, cost):
    import os

    # Ensure file path is next to main.py
    file_path = os.path.join(os.path.dirname(__file__), "cost-analysis.md")
    print("_" * 60)
    print("DEBUG PATH:", os.path.abspath(file_path))

    # Check if file exists or is empty
    need_header = False
    if not os.path.exists(file_path):
        need_header = True
    else:
        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            need_header = True

    # If header is needed, write header first
    if need_header:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("# Cost Analysis — Gemini API Usage\n\n")
            f.write("| ID | Model | Input Tokens | Output Tokens | Total Tokens | Latency (ms) | Estimated Cost ($) |\n")
            f.write("|----|-------|-------------|--------------|--------------|--------------|--------------------|\n")

    # Read existing lines to determine next ID
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Count existing data rows (skip header lines)
    existing_rows = [line for line in lines if line.strip().startswith("|") and line.strip()[1].isdigit()]
    next_id = len(existing_rows) + 1

    # Create new row with model name
    new_row = f"| {next_id} | {model_name} | {input_tokens} | {output_tokens} | {total_tokens} | {latency_ms:.0f} | {cost:.6f} |\n"
    print("DEBUG ROW:", new_row.strip())

    # Append row to file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(new_row)

    print("update finished")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # 1. Get the API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment.")
        print("Make sure you have a .env file with: GEMINI_API_KEY=your_key_here")
        print("See guides/gemini-setup-guide.md for instructions.")
        exit(1)

    # 2. Create the client
    print(f"Connecting to {MODEL}...")
    client = genai.Client(api_key=api_key)

    # 3. Count tokens before generating (optional but educational)
    token_count_result = client.models.count_tokens(
        model=MODEL,
        contents=PROMPT
    )
    print(f"\nPrompt: {PROMPT}")
    print(f"\nExpected input tokens: {token_count_result.total_tokens}")
    print("\nSending request...\n")

    # 4. Make the API call and measure latency
    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=MODEL,
        contents=PROMPT
    )

    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000

    # 5. Display the response
    print("─" * 60)
    print("RESPONSE:")
    print("─" * 60)
    print(response.text)
    print("─" * 60)


    # 6. Display usage metrics
    usage = response.usage_metadata
    print("\nTOKEN USAGE:")
    print(f"  Input tokens:  {usage.prompt_token_count}")
    print(f"  Output tokens: {usage.candidates_token_count}")
    print(f"  Total tokens:  {usage.total_token_count}")

    print(f"\nLATENCY:")
    print(f"  Total time: {latency_ms:.0f} ms")

    print(f"\nCOST ESTIMATE:")
    print(f"  Free tier:  $0.00 (you are not being charged)")
    # Reference calculation for awareness
    input_cost  = (usage.prompt_token_count    / 1_000_000) * 0.10
    output_cost = (usage.candidates_token_count / 1_000_000) * 0.40
    paid_cost   = input_cost + output_cost
    print(f"  Paid tier equivalent (gemini-3-flash-preview): ${paid_cost:.6f}")

    print("\n✓ Exercise 1 complete. Proceed to Exercise 2.")
    print("  File: examples/starter-code/02_prompt_patterns.py")

    
    update_cost_analysis(
        MODEL,
        usage.prompt_token_count,
        usage.candidates_token_count,
        usage.total_token_count,
        latency_ms,
        paid_cost
                )




if __name__ == "__main__":
    main()
