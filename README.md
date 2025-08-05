
=================================================================
What is SynthAgent Arena?

SynthAgent Arena is an interactive command-line application that lets you run multi-agent conversations powered by large language models (LLMs). You can choose between several modes:

- **Business Task:** Agents collaborate to solve real-world business problems.
- **Research:** Agents gather and summarize information for you.
- **War Room:** Four agents with distinct personalities work together to find the best solution to a challenge.
- **Debate:** Two agents debate a topic, with a third agent acting as referee.

Who is it for?
This script is ideal for:
- Developers and researchers experimenting with LLMs and agent-based systems
- Product managers, business analysts, and strategists who want AI-powered brainstorming or decision support
- Anyone interested in building more complex multi-agent frameworks, chatbots, or prompt engineering tools

Why is this a great base?
SynthAgent Arena is designed to be simple, modular, and extensible. It demonstrates:
- How to orchestrate multiple LLM agents in a single workflow
- How to integrate external APIs (OpenAI, Gemini, Ollama, Crawl4AI) and local models
- How to use personas and prompt engineering to create realistic, useful agent behaviors
- How to log, evaluate, and summarize agent interactions

You can use this script as a starting point for larger projects, frameworks, or custom LLM applications. The code is easy to read, well-commented, and built to fail gracefully if a backend is missing. It’s a practical example for anyone learning about multi-agent AI, conversational interfaces, or prompt design.

Tech Overview
- **Python 3.8+**
- **LLM backends:** OpenAI (GPT-4, GPT-3.5), Google Gemini, Ollama (local models like deepseek, llama3)
- **Web research:** Crawl4AI for real-time internet search
- **Terminal UI:** Rich and Pyfiglet for colorful, readable output
- **Extensible:** Add new agent roles, backends, or conversation modes easily

Key features
------------
* **Multi‑Agent conversation:** debates are driven by two large language
  models.  Models can be OpenAI (e.g. `gpt-4.1`, `gpt-4o`), Google
  Gemini (`gemini-pro`) or local models served via `ollama` (e.g.
  `llama3.2`, `deepseek`, etc.).

* **Referee agent:** a separate language model analyses the full
  conversation transcript and declares a winner along with a detailed
  rationale.

* **Topic generation:** topics for debate are generated dynamically
  using OpenAI.  If an API key is not configured the system falls back
  to a predefined list of interesting controversies.

* **Timed and endless modes:** debates can be limited to 60 seconds,
  5 minutes, 10 minutes or allowed to continue until a winner is
  declared (endless mode is ideal for philosophical questions).

* **Web research via Crawl4AI:** agents can optionally consult the
  internet using the open‑source `crawl4ai` crawler for supporting
  evidence.

Before running, install the dependencies (for example via
`pip install openai google-generativeai ollama-python crawl4ai rich pyfiglet`)
and set the required API keys in your environment:

```
export OPENAI_API_KEY="sk-..."        # for OpenAI models and the referee
export GOOGLE_API_KEY="api-key"        # for Gemini (optional)
# ensure `ollama` daemon is running locally with desired models
```

Because this script interacts with external services, it is designed to
fail gracefully. If a particular backend isn’t configured the script
will inform the user and skip the related functionality

---

## License

SynthAgent Arena is open source software released under the MIT License.

Copyright (c) 2025 Jeremy Harris of Augments | augments.art

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---
