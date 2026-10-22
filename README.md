# Memory Garden

**An agent that decides what to remember out loud.** Strands Decider 2B judges every stored memory in about 100 ms, so Amazon Bedrock only sees the few memories that matter for the current message.

Built for the **AWS Agent Experience Hackathon: Strands Decider**.
Tags: `#strands-decider-2b` `#context-memory`

## The problem

Long-running agents pile up memories. Pasting all of them into every prompt wastes tokens, leaks private details, and confuses the model with outdated facts ("is vegetarian" next to "eats fish again"). Asking an LLM to filter them adds latency and cost to every turn.

## The decision handed to Decider 2B

For each memory chunk, one read of Decider 2B answers four typed yes/no (`noul`) questions, each with a calibrated probability:

| Question | Purpose |
|---|---|
| `relevant` | Does this memory help answer the user's message? |
| `stale` | Is it outdated and no longer true? |
| `superseded` | Do newer memories on the same topic contradict it? |
| `private` | Is it sensitive (health, money, credentials) and unnecessary here? |

Plain Python then turns the probabilities into a verdict (see `verdict()` in `app/main.py`):

- `withheld`: private and not clearly needed
- `pruned`: stale or superseded
- `injected`: relevant enough (threshold is adjustable)
- `skipped`: not needed

Only `injected` memories go into the Bedrock system prompt. Decider 2B does not generate text; Bedrock writes the answer.

## What you see

- **Garden view:** every memory is a bloom. Gold blooms are used, red are outdated, purple are private, dim are not needed. Hover for the four probabilities.
- **Pipeline bar, scan wave, golden threads** that show the decision flow live.
- **Decisions tab:** a ledger of every verdict with relevance and latency.
- **Live threshold slider:** re-applies the decision rule instantly, without calling the model again.
- **Token bar:** memory tokens sent to the model vs all stored memory tokens.

## Architecture

```
user message -> FastAPI /api/ask (SSE stream)
   -> for each memory: Decider 2B (/v1/systemone, 4 noul questions)   [local]
   -> verdict() rule over the probabilities
   -> injected memories -> Amazon Bedrock ConverseStream (Nova Lite)   [AWS]
   -> tokens streamed back to the garden UI
```

## Run it

Requires Python 3.11 or 3.12 (3.13 may lack PyTorch wheels), an AWS account with Bedrock model access for Amazon Nova Lite, and credentials for real answers.

**Terminal 1: Decider server**
```bash
python3.12 -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
pip install strands-decider
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8099
```

**Terminal 2: app**
```bash
# activate the same venv
pip install -r requirements.txt
# PowerShell: $env:AWS_REGION="us-east-1"; $env:AWS_ACCESS_KEY_ID="..."; $env:AWS_SECRET_ACCESS_KEY="..."
uvicorn app.main:app --port 8000
```
Open http://localhost:8000.

**Demo mode (no Bedrock call):** set `MOCK_LLM=1` before `uvicorn`. The Decider gate and the whole UI run for real; only the final answer is canned.

**Fallback:** if the Decider server is unreachable, the app uses a labelled keyword heuristic. That is not the real model.

Configuration: see `.env.example` (`DECIDER_URL`, `AWS_REGION`, `BEDROCK_MODEL_ID`). Model IDs can differ by region; use an inference-profile ID if your region requires one.

Quick check of the Decider response shape (PowerShell):
```powershell
$body = @{ state="I love tea"; questions=@{ relevant=@{ type="noul"; instructions="Is this about drinks?" } } } | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://localhost:8099/v1/systemone -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```
The answer is read from `answers.<name>.noul`.

## Reusable decision pattern: Context Gate

Copy these three pieces from `app/main.py` into any agent's pre-model step:

1. `QUESTIONS`: the four typed `noul` questions above.
2. `state_for(query, memory)`: builds the state text (message, memory with its age and topic, newer memories on the same topic).
3. `verdict(probabilities, threshold)`: the plain-Python rule that picks inject, prune, withhold, or skip.

Wiring: call the gate before the model call, then build the system prompt from the `injected` memories only. Tune the questions and the thresholds on your own data; they were picked by hand.

## Limitations

- Decider 2B is an experimental model. Its confidence is calibrated on short classification tasks, so measure on your own memories before trusting a threshold.
- The seed memories are a small hand-written demo set.
- Scoring runs 4 chunks in parallel against one local server, so latency grows with the memory count. On CPU a full scan takes longer than on GPU.
- No latency or cost benchmark is included yet. Do not treat the token bar as a measured production saving; it estimates tokens as characters divided by four.

## Security

Never commit AWS keys or a `.env` file. Use environment variables or `aws configure`.

## Credits

Built on [Strands Decider](https://github.com/strands-labs/strands-decider) (Apache-2.0) and Amazon Bedrock.
