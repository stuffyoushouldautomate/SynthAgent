import asyncio
import json
import os
import random
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

# Optional third‑party imports wrapped in try/except so that missing
# packages don’t prevent the script from starting.  The script
# notifies the user when a backend is unavailable.
try:
    import requests  # type: ignore
except ImportError:
    requests = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:
    openai = None  # type: ignore

try:
    import google.generativeai as genai  # type: ignore
except ImportError:
    genai = None  # type: ignore

try:
    import ollama  # type: ignore
except ImportError:
    ollama = None  # type: ignore

try:
    from crawl4ai import AsyncWebCrawler  # type: ignore
except ImportError:
    AsyncWebCrawler = None  # type: ignore

try:
    from rich.console import Console
    from rich.table import Table
    from rich.prompt import Prompt
    from rich.panel import Panel
    from rich.align import Align
    from rich.style import Style
except ImportError:
    sys.stderr.write(
        "[warning] rich library is not installed; install it with `pip install rich`\n"
    )
    Console = None  # type: ignore
    Table = None  # type: ignore
    Prompt = None  # type: ignore
    Panel = None  # type: ignore
    Align = None  # type: ignore
    Style = None  # type: ignore

try:
    from pyfiglet import Figlet
except ImportError:
    Figlet = None  # type: ignore


def print_banner(console: 'Console') -> None:
    """Render the application banner using pyfiglet and rich for style."""
    if Figlet:
        fig = Figlet(font="slant")
        banner = fig.renderText("SynthAgent Arena")
        console.print(banner, style=Style(color="cyan", bold=True))
    else:
        console.print("=== SynthAgent Arena ===", style=Style(color="cyan", bold=True))
    console.print(
        "Welcome to SynthAgent Arena — a head‑to‑head battle of wits powered by AI!",
        style="magenta",
    )
    console.print(
        "Select a topic, choose your champions and watch them debate with real‑time research.\n",
        style="magenta",
    )


def get_available_backends() -> Dict[str, bool]:
    """Return a dictionary indicating which model backends are available."""
    backends = {
        "openai": requests is not None and os.getenv("OPENAI_API_KEY") is not None,
        "gemini": genai is not None and os.getenv("GOOGLE_API_KEY") is not None,
        "ollama": ollama is not None,
    }
    return backends


def generate_topics_via_openai() -> Optional[List[str]]:
    """Use OpenAI to generate five interesting debate topics.

    Returns a list of topics or None if generation fails.
    """
    if requests is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        input_messages = [
            {
                "role": "system",
                "content": "You are a creative assistant that generates thought‑provoking debate topics. Focus on current, relevant, and controversial issues that will spark engaging discussions."
            },
            {
                "role": "user",
                "content": (
                    "Generate a numbered list of five concise yet intriguing debate topics "
                    "covering technology, philosophy, science, culture, ethics, and current events. "
                    "Ensure they are clear, distinct, contentious enough to spark lively discussion, "
                    "and relevant to today's world. Avoid generic topics - make them specific and thought-provoking. "
                    "If possible, reference current events, recent developments, or emerging trends. "
                    "Format as: 1. [Topic] 2. [Topic] etc."
                ),
            },
        ]
        
        data = {
            "model": "gpt-4.1",
            "input": input_messages,
            "text": {
                "format": {
                    "type": "text"
                }
            },
            "reasoning": {},
            "tools": [],
            "temperature": 0.9,
            "max_output_tokens": 300,
            "top_p": 1,
            "store": True
        }
        
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return None
            
        result = response.json()
        # The new API returns content in output[0].content[0].text
        if "output" in result and len(result["output"]) > 0:
            text = result["output"][0]["content"][0]["text"].strip()
        else:
            return None
        # Extract lines beginning with a digit
        topics = []
        for line in text.splitlines():
            line = line.strip()
            if line and line[0].isdigit():
                # Remove the leading number and punctuation
                topic = line.lstrip("0123456789").lstrip(". ")
                topics.append(topic)
        if len(topics) >= 5:
            return topics[:5]
        return None
    except Exception as e:
        # Swallow exceptions quietly; fallback will be used instead
        return None


def get_custom_topic(console: 'Console') -> str:
    """Prompt the user to enter their own debate topic."""
    console.print("\n🎯 Enter Your Own Debate Topic", style="bold cyan")
    console.print("Type your debate topic below. Make it specific and thought-provoking!", style="cyan")
    
    topic = Prompt.ask("Your topic")
    if not topic.strip():
        console.print("⚠️  No topic entered, using a generated topic instead.", style="yellow")
        return None
    
    return topic.strip()


def get_debate_topics() -> List[str]:
    """Get five debate topics via OpenAI. Falls back to static list only if OpenAI is completely unavailable."""
    # Always try OpenAI first for fresh, random topics
    topics = generate_topics_via_openai()
    if topics:
        return topics
    
    # Only fall back to static topics if OpenAI is completely unavailable
    # This ensures we get fresh topics whenever possible
    if requests is None or os.getenv("OPENAI_API_KEY") is None:
        print("⚠️  OpenAI not available - using fallback topics")
        return [
            "Should artificial general intelligence be open sourced?",
            "Is space colonisation the right solution to Earth's environmental crises?",
            "Can machines ever truly be conscious?",
            "Does social media enhance or erode democracy?",
            "Is nuclear energy the most viable path to a carbon‑neutral future?",
        ]
    else:
        # If OpenAI is available but topic generation failed, try one more time
        print("🔄 Retrying topic generation...")
        topics = generate_topics_via_openai()
        if topics:
            return topics
        else:
            print("⚠️  Topic generation failed - using fallback topics")
            return [
                "Should artificial general intelligence be open sourced?",
                "Is space colonisation the right solution to Earth's environmental crises?",
                "Can machines ever truly be conscious?",
                "Does social media enhance or erode democracy?",
                "Is nuclear energy the most viable path to a carbon‑neutral future?",
            ]


async def crawl_research(topic: str) -> Optional[str]:
    """Use Crawl4AI to gather quick research on a topic.

    Returns markdown content or None.  Requires AsyncWebCrawler from crawl4ai.
    """
    if AsyncWebCrawler is None:
        return None
    try:
        async with AsyncWebCrawler() as crawler:
            # Search for recent information about the topic
            search_query = f"latest news debate {topic}"
            result = await crawler.arun(url=f"https://www.google.com/search?q={search_query}")
            return result.markdown if hasattr(result, "markdown") else None
    except Exception:
        return None


async def crawl_additional_research(query: str) -> Optional[str]:
    """Use Crawl4AI to gather additional research on a specific query."""
    if AsyncWebCrawler is None:
        return None
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=f"https://www.google.com/search?q={query}")
            return result.markdown if hasattr(result, "markdown") else None
    except Exception:
        return None


async def scrape_current_headlines() -> Optional[List[str]]:
    """Use Crawl4AI to scrape current headlines from US news sites.
    
    Returns a list of current headlines or None if scraping fails.
    """
    # Just use the simple scraping method - let OpenAI handle the processing
    return await scrape_simple_headlines()


async def scrape_simple_headlines() -> Optional[List[str]]:
    """Use Crawl4AI to scrape headlines directly from news sites.
    
    Returns a list of current headlines or None if scraping fails.
    """
    if AsyncWebCrawler is None:
        return None
    
    headlines = []
    # Major US news sites that are reliable for scraping
    news_sites = [
        "https://www.bbc.com/news/world/us_and_canada",
        "https://www.reuters.com/world/us",
        "https://www.theguardian.com/us-news",
        "https://www.cnn.com/us",
        "https://www.nbcnews.com/us-news"
    ]
    
    try:
        async with AsyncWebCrawler() as crawler:
            for site in news_sites[:3]:  # Limit to 3 sites
                try:
                    result = await crawler.arun(url=site)
                    if result.markdown:
                        # Look for actual news headlines in the content
                        lines = result.markdown.split('\n')
                        for line in lines:
                            line = line.strip()
                            # Look for lines that look like news headlines
                            if (len(line) > 30 and len(line) < 150 and  # Reasonable length
                                any(word in line.lower() for word in ['breaking', 'news', 'update', 'announces', 'launches', 'reports', 'says', 'reveals', 'confirms', 'president', 'election', 'court', 'police', 'government', 'congress', 'senate', 'house', 'biden', 'trump', 'democrat', 'republican', 'campaign', 'vote', 'law', 'justice', 'federal', 'state', 'city', 'mayor', 'governor', 'senator', 'representative']) and
                                not any(skip_word in line.lower() for skip_word in ['skip to', 'menu', 'search', 'subscribe', 'advertisement', 'cookie', 'privacy', 'terms', 'contact', 'about', 'help', 'sign in', 'watch live', 'home', 'news', 'sport', 'weather', 'iplayer', 'sounds', 'bitesize', 'cbeebies', 'cbbc']) and
                                not line.startswith('[') and not line.endswith(']') and  # Skip markdown links
                                not line.startswith('http') and not line.endswith('.com') and  # Skip URLs
                                not any(char in line for char in ['<', '>', '(', ')', '{', '}'])):  # Skip HTML/XML
                                # Clean up the headline
                                headline = line.replace('#', '').replace('*', '').strip()
                                if headline and headline not in headlines:  # Avoid duplicates
                                    headlines.append(headline)
                                    if len(headlines) >= 5:  # Get 5 headlines
                                        break
                except Exception as e:
                    continue  # Skip this site if it fails
                
                if len(headlines) >= 5:
                    break
                    
        return headlines[:5] if headlines else None
        
    except Exception as e:
        return None


async def get_current_events_context() -> Optional[str]:
    """Get current events context for debate topics."""
    headlines = await scrape_simple_headlines()
    if headlines:
        context = "**Current Events Context:**\n\n"
        for i, headline in enumerate(headlines, 1):
            context += f"{i}. {headline}\n"
        context += "\nConsider these current events when formulating your arguments."
        return context
    
    # If no headlines found, return None instead of fallback
    return None


async def display_current_headlines(console: 'Console') -> None:
    """Display current headlines to the user."""
    if AsyncWebCrawler is None:
        return
    
    console.print("\n📰 Current Headlines", style="bold cyan")
    console.print("Gathering latest news...", style="dim")
    
    headlines = await scrape_simple_headlines()
    if headlines:
        console.print(Panel.fit(
            "\n".join([f"• {headline}" for headline in headlines]),
            title="📰 Latest Headlines",
            style="bold blue"
        ))
        console.print("These current events will inform the debate topics and arguments.", style="cyan")
    else:
        console.print("⚠️  No current headlines found. Debates will proceed without current events context.", style="yellow")


async def generate_agent_persona(name: str, topic: str) -> Dict[str, str]:
    """Generate a full persona for an agent including backstory, personality, and debate motivation."""
    if requests is None:
        return {
            "name": name,
            "backstory": f"{name} is a passionate debater with strong opinions.",
            "personality": "Direct and argumentative",
            "demographics": "Middle-aged, educated",
            "likes": "Intellectual discussions",
            "dislikes": "Weak arguments",
            "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
            "confidence_level": "Very confident",
            "debate_style": "Aggressive and direct"
        }
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {
            "name": name,
            "backstory": f"{name} is a passionate debater with strong opinions.",
            "personality": "Direct and argumentative",
            "demographics": "Middle-aged, educated",
            "likes": "Intellectual discussions",
            "dislikes": "Weak arguments",
            "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
            "confidence_level": "Very confident",
            "debate_style": "Aggressive and direct"
        }
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        input_messages = [
            {
                "role": "system",
                "content": "You are a creative writer who creates realistic, detailed personas for debate participants. Make them feel like real people with complex backgrounds, personalities, and motivations."
            },
            {
                "role": "user",
                "content": (
                    f"Create a detailed persona for a debate participant named '{name}' who will be debating the topic: '{topic}'. "
                    f"Generate a realistic, complex character with:\n"
                    f"1. A detailed backstory (where they grew up, education, career, life experiences)\n"
                    f"2. Personality traits (how they think, behave, communicate)\n"
                    f"3. Demographics (age, background, socioeconomic status, location)\n"
                    f"4. Likes and dislikes (personal preferences, pet peeves, interests)\n"
                    f"5. Debate motivation (why they think they're uniquely qualified for this topic)\n"
                    f"6. Confidence level (how confident they are about their position)\n"
                    f"7. Debate style (how they argue, communicate, handle conflict)\n"
                    f"8. Emotional state (how they feel about the topic, their opponent, the debate)\n"
                    f"9. Personal stakes (what's at stake for them personally in this debate)\n"
                    f"10. Communication quirks (how they speak, any catchphrases, mannerisms)\n\n"
                    f"Make them feel like a real person with flaws, biases, and authentic motivations. "
                    f"They should be capable of both intellectual insight and petty behavior. "
                    f"Return the response as a JSON object with these exact keys: backstory, personality, demographics, likes, dislikes, debate_motivation, confidence_level, debate_style, emotional_state, personal_stakes, communication_quirks"
                ),
            },
        ]
        
        data = {
            "model": "gpt-4.1",
            "input": input_messages,
            "text": {
                "format": {
                    "type": "text"
                }
            },
            "reasoning": {},
            "tools": [],
            "temperature": 0.9,
            "max_output_tokens": 800,
            "top_p": 1,
            "store": True
        }
        
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return {
                "name": name,
                "backstory": f"{name} is a passionate debater with strong opinions.",
                "personality": "Direct and argumentative",
                "demographics": "Middle-aged, educated",
                "likes": "Intellectual discussions",
                "dislikes": "Weak arguments",
                "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
                "confidence_level": "Very confident",
                "debate_style": "Aggressive and direct"
            }
            
        result = response.json()
        if "output" in result and len(result["output"]) > 0:
            text = result["output"][0]["content"][0]["text"].strip()
            try:
                # Try to parse as JSON
                persona = json.loads(text)
                persona["name"] = name
                return persona
            except json.JSONDecodeError:
                # If not valid JSON, create a basic persona
                return {
                    "name": name,
                    "backstory": f"{name} is a passionate debater with strong opinions.",
                    "personality": "Direct and argumentative",
                    "demographics": "Middle-aged, educated",
                    "likes": "Intellectual discussions",
                    "dislikes": "Weak arguments",
                    "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
                    "confidence_level": "Very confident",
                    "debate_style": "Aggressive and direct"
                }
        return {
            "name": name,
            "backstory": f"{name} is a passionate debater with strong opinions.",
            "personality": "Direct and argumentative",
            "demographics": "Middle-aged, educated",
            "likes": "Intellectual discussions",
            "dislikes": "Weak arguments",
            "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
            "confidence_level": "Very confident",
            "debate_style": "Aggressive and direct"
        }
    except Exception:
        return {
            "name": name,
            "backstory": f"{name} is a passionate debater with strong opinions.",
            "personality": "Direct and argumentative",
            "demographics": "Middle-aged, educated",
            "likes": "Intellectual discussions",
            "dislikes": "Weak arguments",
            "debate_motivation": f"{name} believes they are uniquely qualified to discuss this topic due to their experience.",
            "confidence_level": "Very confident",
            "debate_style": "Aggressive and direct"
        }


async def generate_agent_names() -> Tuple[str, str]:
    """Generate funny vintage agent names using AI."""
    if requests is None:
        return "Randy", "Earl"
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "Randy", "Earl"
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        input_messages = [
            {
                "role": "system",
                "content": "You are a creative assistant that generates funny, vintage, classic names for AI debate agents. Use old-fashioned, vintage names that are memorable and humorous."
            },
            {
                "role": "user",
                "content": (
                    "Generate two funny vintage names for AI debate agents. "
                    "Use classic, old-fashioned names that are humorous and memorable. "
                    "Examples: Randy, Earl, Mabel, Henrietta, Gertrude, Clarence, Mildred, Herbert, Ethel, Wilbur, Agnes, Otis, Bertha, Virgil, Myrtle, etc. "
                    "Make them sound like characters from an old sitcom or classic TV show. "
                    "Return only the two names, one per line, no numbers or formatting."
                ),
            },
        ]
        
        data = {
            "model": "gpt-4.1",
            "input": input_messages,
            "text": {
                "format": {
                    "type": "text"
                }
            },
            "reasoning": {},
            "tools": [],
            "temperature": 0.9,
            "max_output_tokens": 100,
            "top_p": 1,
            "store": True
        }
        
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            return "Randy", "Earl"
            
        result = response.json()
        if "output" in result and len(result["output"]) > 0:
            text = result["output"][0]["content"][0]["text"].strip()
            names = [name.strip() for name in text.split('\n') if name.strip()]
            if len(names) >= 2:
                return names[0], names[1]
        return "Randy", "Earl"
    except Exception:
        return "Randy", "Earl"


def auto_select_models(backends: Dict[str, bool]) -> Tuple[str, str, str, str]:
    """Auto-select the best available models for each backend."""
    # Default to Ollama deepseek-r1:8b if available
    if backends.get("ollama"):
        return "ollama", "deepseek-r1:8b", "ollama", "deepseek-r1:8b"
    # Otherwise, fallback to OpenAI or Gemini
    available_backends = [name for name, available in backends.items() if available]
    if len(available_backends) < 2:
        if available_backends:
            backend = available_backends[0]
            if backend == "openai":
                return backend, "gpt-4.1", backend, "gpt-4.1"
            elif backend == "gemini":
                return backend, "gemini-1.5-pro", backend, "gemini-1.5-pro"
        else:
            return "openai", "gpt-4.1", "openai", "gpt-4.1"
    backend1 = available_backends[0]
    backend2 = available_backends[1] if len(available_backends) > 1 else available_backends[0]
    model1 = "gpt-4.1" if backend1 == "openai" else "gemini-1.5-pro" if backend1 == "gemini" else "llama3"
    model2 = "gpt-4.1" if backend2 == "openai" else "gemini-1.5-pro" if backend2 == "gemini" else "llama3"
    return backend1, model1, backend2, model2


def choose_debate_mode(console) -> Tuple[str, bool]:
    """Let user choose between generated topics, custom topic, and debate mode."""
    console.print("\n🎯 Choose Your Debate Mode", style="bold cyan")
    console.print("1. Use generated topics (AI creates 5 random topics)")
    console.print("2. Enter your own custom topic")
    console.print("3. Interactive mode (you can interrupt and add info)")
    console.print("4. Standard mode (no interruptions)")
    
    mode = Prompt.ask("Select mode", choices=["1", "2", "3", "4"], default="1")
    
    if mode == "2":
        # Custom topic mode
        custom_topic = get_custom_topic(console)
        if custom_topic:
            return custom_topic, True  # Allow interruptions for custom topics
        else:
            console.print("Using generated topics instead.", style="yellow")
            return None, False
    elif mode == "3":
        # Interactive mode
        return None, True
    else:
        # Standard mode
        return None, False


async def call_openai(model: str, messages: List[Dict[str, str]]) -> str:
    """Send chat messages to an OpenAI model and return the assistant's reply."""
    if requests is None:
        raise RuntimeError("Requests library is not available.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    
    # Convert messages to input format for responses API
    input_messages = []
    for msg in messages:
        input_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "input": input_messages,
        "text": {
            "format": {
                "type": "text"
            }
        },
        "reasoning": {},
        "tools": [],
        "temperature": 0.7,
        "max_output_tokens": 512,
        "top_p": 1,
        "store": True
    }
    
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=data
    )
    
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
    
    result = response.json()
    # The new API returns content in output[0].content[0].text
    if "output" in result and len(result["output"]) > 0:
        content = result["output"][0]["content"][0]["text"]
        return content.strip()
    else:
        raise RuntimeError("Unexpected response format from OpenAI API")


async def call_gemini(model: str, messages: List[Dict[str, str]]) -> str:
    """Send chat messages to a Gemini model and return the reply."""
    if genai is None:
        raise RuntimeError("Gemini backend is not available.")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    # Gemini expects one string prompt; combine messages
    prompt_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            # System messages can be implicitly included in the instruction
            prompt_parts.append(f"(System note: {content})")
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        else:
            prompt_parts.append(f"Assistant: {content}")
    prompt = "\n".join(prompt_parts)
    # Use generative model; gemini-pro supports multi-turn chat via .start_chat()
    model_obj = genai.GenerativeModel(model)
    chat = model_obj.start_chat(history=[])
    response = chat.send_message(prompt)
    return response.text.strip()


async def call_ollama(model: str, messages: List[Dict[str, str]]) -> str:
    """Send chat messages to a model served by Ollama and return the reply."""
    if ollama is None:
        raise RuntimeError("Ollama backend is not available.")
    # Ollama expects messages in the same format as OpenAI
    response = ollama.chat(model=model, messages=messages)
    # Response is a dict with 'message' key containing dict with 'content'
    return response["message"]["content"].strip()


async def get_model_reply(backend: str, model: str, messages: List[Dict[str, str]]) -> str:
    """Dispatch to the appropriate backend function based on backend name."""
    if backend == "openai":
        return await call_openai(model, messages)
    if backend == "gemini":
        return await call_gemini(model, messages)
    if backend == "ollama":
        return await call_ollama(model, messages)
    raise ValueError(f"Unsupported backend: {backend}")


async def evaluate_debate(transcript: List[Tuple[str, str]], topic: str) -> Tuple[str, str]:
    """Ask OpenAI to judge the winner of a debate and provide a rationale.

    Returns a tuple of (winner_name, explanation).  The transcript should be
    a list of (speaker_name, utterance) tuples.  The function uses
    gpt‑4.1 for evaluation.
    """
    if requests is None:
        raise RuntimeError("Referee evaluation requires requests library.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")
    
    # Build the conversation transcript as a string
    conv = []
    for speaker, utterance in transcript:
        conv.append(f"{speaker}: {utterance}")
    conversation_str = "\n".join(conv)
    system_prompt = (
        "You are a meticulous debate judge.  Analyse the following debate transcript "
        "between two agents discussing the topic '{topic}'.  Consider clarity, evidence, "
        "logic, rhetorical skill and adherence to the topic.  Identify which agent "
        "presented the stronger case and explain your reasoning in detail.  "
        "Reply with JSON using the format: {{'winner': 'AGENT_NAME', 'reason': '<detailed rationale>'}}."
    ).format(topic=topic)
    
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": conversation_str},
    ]
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4.1",
        "input": input_messages,
        "text": {
            "format": {
                "type": "text"
            }
        },
        "reasoning": {},
        "tools": [],
        "temperature": 0.0,
        "max_output_tokens": 900,
        "top_p": 1,
        "store": True
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
            
        result = response.json()
        # The new API returns content in output[0].content[0].text
        if "output" in result and len(result["output"]) > 0:
            content = result["output"][0]["content"][0]["text"].strip()
        else:
            raise RuntimeError("Unexpected response format from OpenAI API")
    except Exception as e:
        return "Unknown", f"Evaluation failed: {e}"
    
    try:
        data = json.loads(content)
        return data.get("winner", "Unknown"), data.get("reason", "")
    except json.JSONDecodeError:
        # If the model did not return JSON, treat the whole response as explanation
        return "Unknown", content


async def handle_user_interruption(console, topic: str, messages1: List[Dict[str, str]], messages2: List[Dict[str, str]], name1: str, name2: str, debate_id: str) -> bool:
    """Handle user interruption during debate. Returns True if user wants to continue, False to end."""
    console.print("\n🎤 User Interruption Mode", style="bold magenta")
    console.print("You can now add information, redirect the conversation, or ask questions.", style="magenta")
    console.print("Type your input (or press Enter to continue the debate):", style="cyan")
    
    user_input = input().strip()
    if not user_input:
        return True  # Continue debate
    
    # Log user interruption
    logs_dir = create_debate_log_directory()
    interruption_log = os.path.join(logs_dir, f"interruption_{debate_id}.md")
    
    try:
        with open(interruption_log, 'a', encoding='utf-8') as f:
            f.write(f"## 🎤 User Interruption - {datetime.now().strftime('%H:%M:%S')}\n\n")
            f.write(f"**User Input:** {user_input}\n\n")
    except Exception:
        pass  # Don't fail if logging fails
    
    # Check if user wants to add web research
    if user_input.lower().startswith("research:"):
        query = user_input[9:].strip()
        if query:
            console.print(f"🔍 Researching: {query}", style="yellow")
            research_result = await crawl_additional_research(query)
            if research_result:
                user_message = f"User provided research on '{query}': {research_result}"
                console.print(Panel.fit(research_result[:200] + "...", title="🔍 Research Results", style="bold yellow"))
                
                # Log research result
                try:
                    with open(interruption_log, 'a', encoding='utf-8') as f:
                        f.write(f"**Research Query:** {query}\n")
                        f.write(f"**Research Result:** {research_result[:500]}...\n\n")
                except Exception:
                    pass
            else:
                user_message = f"User requested research on '{query}' but no results found."
                console.print("❌ No research results found.", style="red")
        else:
            user_message = f"User interruption: {user_input}"
    else:
        user_message = f"User interruption: {user_input}"
    
    # Add user input to both agents' conversation histories
    messages1.append({"role": "user", "content": user_message})
    messages2.append({"role": "user", "content": user_message})
    
    # Display user input
    console.print(Panel.fit(user_input, title="🎤 User Input", style="bold magenta"))
    
    # Ask if user wants to continue or end
    console.print("\nOptions:", style="bold")
    console.print("1. Continue debate (agents will respond to your input)")
    console.print("2. End debate now")
    console.print("3. Add more information")
    console.print("4. Research something specific (type 'research: your query')")
    
    choice = Prompt.ask("Choose option", choices=["1", "2", "3", "4"], default="1")
    
    # Log user choice
    try:
        with open(interruption_log, 'a', encoding='utf-8') as f:
            f.write(f"**User Choice:** {choice}\n\n---\n\n")
    except Exception:
        pass
    
    if choice == "2":
        return False  # End debate
    elif choice == "3":
        # Recursive call to add more information
        return await handle_user_interruption(console, topic, messages1, messages2, name1, name2, debate_id)
    elif choice == "4":
        console.print("Type 'research: your query' to search for specific information.", style="cyan")
        return await handle_user_interruption(console, topic, messages1, messages2, name1, name2, debate_id)
    else:
        return True  # Continue debate


async def conduct_debate(
    topic: str,
    backend1: str,
    model1: str,
    name1: str,
    backend2: str,
    model2: str,
    name2: str,
    time_limit_seconds: Optional[int] = None,
    allow_interruptions: bool = False,
) -> None:
    """Manage the debate loop between two agents and display results.

    Parameters
    ----------
    topic : str
        The topic of the debate.
    backend1, backend2 : str
        The names of the backends ('openai', 'gemini', 'ollama') for each agent.
    model1, model2 : str
        The specific model names (e.g. 'gpt-4', 'deepseek-r1:8b', 'gemini-pro', 'llama3').
    name1, name2 : str
        Friendly display names for the agents.
    time_limit_seconds : Optional[int]
        If provided, the maximum duration of the debate in seconds.  None means unlimited.
    allow_interruptions : bool
        Whether to allow user interruptions during the debate.
    """
    console = Console() if Console else None
    
    # Generate unique debate ID
    debate_id = str(uuid.uuid4())[:8]
    console.print(f"🎯 Debate ID: {debate_id}", style="dim") if console else None
    
    # Generate personas for each agent
    persona1 = await generate_agent_persona(name1, topic)
    persona2 = await generate_agent_persona(name2, topic)
    
    if console:
        console.print(f"\n🎭 {name1}'s Persona:", style="bold blue")
        console.print(f"   Backstory: {persona1.get('backstory', 'N/A')}")
        console.print(f"   Personality: {persona1.get('personality', 'N/A')}")
        console.print(f"   Motivation: {persona1.get('debate_motivation', 'N/A')}")
        
        console.print(f"\n🎭 {name2}'s Persona:", style="bold green")
        console.print(f"   Backstory: {persona2.get('backstory', 'N/A')}")
        console.print(f"   Personality: {persona2.get('personality', 'N/A')}")
        console.print(f"   Motivation: {persona2.get('debate_motivation', 'N/A')}")
    
    # Compose system prompts with personas
    system_prompt_agent1 = (
        f"You are {name1}, a real person participating in a live debate about '{topic}'. "
        f"Your persona: {persona1.get('backstory', '')} "
        f"Personality: {persona1.get('personality', '')} "
        f"Demographics: {persona1.get('demographics', '')} "
        f"Likes: {persona1.get('likes', '')} "
        f"Dislikes: {persona1.get('dislikes', '')} "
        f"Debate motivation: {persona1.get('debate_motivation', '')} "
        f"Confidence level: {persona1.get('confidence_level', '')} "
        f"Debate style: {persona1.get('debate_style', '')} "
        f"Emotional state: {persona1.get('emotional_state', '')} "
        f"Personal stakes: {persona1.get('personal_stakes', '')} "
        f"Communication quirks: {persona1.get('communication_quirks', '')} "
        f"\n\nYou are debating {name2}, who is also a real person. "
        f"This is a realistic debate environment - you can interrupt, use profanity if it fits your character, "
        f"be petty, emotional, or intellectual as your persona dictates. "
        f"Stay completely in character. You believe this is a real debate with a real person. "
        f"Present your arguments clearly and persuasively, but always maintain your unique personality and perspective."
    )
    
    system_prompt_agent2 = (
        f"You are {name2}, a real person participating in a live debate about '{topic}'. "
        f"Your persona: {persona2.get('backstory', '')} "
        f"Personality: {persona2.get('personality', '')} "
        f"Demographics: {persona2.get('demographics', '')} "
        f"Likes: {persona2.get('likes', '')} "
        f"Dislikes: {persona2.get('dislikes', '')} "
        f"Debate motivation: {persona2.get('debate_motivation', '')} "
        f"Confidence level: {persona2.get('confidence_level', '')} "
        f"Debate style: {persona2.get('debate_style', '')} "
        f"Emotional state: {persona2.get('emotional_state', '')} "
        f"Personal stakes: {persona2.get('personal_stakes', '')} "
        f"Communication quirks: {persona2.get('communication_quirks', '')} "
        f"\n\nYou are debating {name1}, who is also a real person. "
        f"This is a realistic debate environment - you can interrupt, use profanity if it fits your character, "
        f"be petty, emotional, or intellectual as your persona dictates. "
        f"Stay completely in character. You believe this is a real debate with a real person. "
        f"Present your arguments clearly and persuasively, but always maintain your unique personality and perspective."
    )
    
    # Initialize conversation histories for each agent
    messages1: List[Dict[str, str]] = [{"role": "system", "content": system_prompt_agent1}]
    messages2: List[Dict[str, str]] = [{"role": "system", "content": system_prompt_agent2}]
    transcript: List[Tuple[str, str]] = []
    start_time = time.time()
    turn = 0
    # Optionally pull research once at start
    research_snippet: Optional[str] = None
    current_events_context: Optional[str] = None
    
    if AsyncWebCrawler is not None:
        try:
            # Get topic-specific research
            research_snippet = await crawl_research(topic)
            # Get current events context
            current_events_context = await get_current_events_context()
        except Exception:
            research_snippet = None
            current_events_context = None
    
    # Combine research and current events
    combined_research = ""
    if research_snippet:
        combined_research += f"Topic Research:\n{research_snippet}\n\n"
    if current_events_context:
        combined_research += f"{current_events_context}\n\n"
    
    if combined_research:
        # Add research snippet as a system note for each agent
        research_note = (
            "Here is some background information and current events that may assist in your arguments:\n\n"
            + combined_research
        )
        messages1.append({"role": "system", "content": research_note})
        messages2.append({"role": "system", "content": research_note})
        if console:
            if research_snippet and current_events_context:
                console.print(Panel.fit(
                    "Research and current events have been provided to both agents.", 
                    title="Research & Current Events", 
                    subtitle="Crawl4AI"
                ))
            elif research_snippet:
                console.print(Panel.fit(
                    "Topic research has been provided to both agents.", 
                    title="Research", 
                    subtitle="Crawl4AI"
                ))
            elif current_events_context:
                console.print(Panel.fit(
                    "Current events context has been provided to both agents.", 
                    title="Current Events", 
                    subtitle="Crawl4AI"
                ))
    # Debate loop
    while True:
        current_time = time.time()
        if time_limit_seconds is not None and current_time - start_time >= time_limit_seconds:
            break
        turn += 1
        
        # Check for user interruption if enabled
        if allow_interruptions and turn > 1:  # Allow interruption after first exchange
            console.print("\n💡 Press Enter to continue, or type something to interrupt...", style="dim")
            try:
                # Non-blocking input check (simplified for now)
                import select
                import sys
                if select.select([sys.stdin], [], [], 0.1)[0]:  # Check if input is available
                    should_continue = await handle_user_interruption(console, topic, messages1, messages2, name1, name2, debate_id)
                    if not should_continue:
                        break
            except:
                pass  # Continue if input check fails
        
        # Agent1 speaks
        user_prompt_1 = "It is your turn to argue. Remember to stay in character and respond as your persona would naturally speak."
        messages1.append({"role": "user", "content": user_prompt_1})
        reply1 = await get_model_reply(backend1, model1, messages1)
        messages1.append({"role": "assistant", "content": reply1})
        messages2.append({"role": "user", "content": reply1})  # pass along to the other agent
        transcript.append((name1, reply1))
        if console:
            console.print(Panel.fit(reply1, title=name1, style="bold blue"))
        
        # Check time limit after each reply
        if time_limit_seconds is not None and time.time() - start_time >= time_limit_seconds:
            break
            
        # Agent2 speaks
        user_prompt_2 = "Respond to your opponent's argument and present your counterpoints. Remember to stay in character and respond as your persona would naturally speak."
        messages2.append({"role": "user", "content": user_prompt_2})
        reply2 = await get_model_reply(backend2, model2, messages2)
        messages2.append({"role": "assistant", "content": reply2})
        messages1.append({"role": "user", "content": reply2})  # pass along
        transcript.append((name2, reply2))
        if console:
            console.print(Panel.fit(reply2, title=name2, style="bold green"))
        
        # In endless mode, evaluate after every 6 turns (3 exchanges each)
        if time_limit_seconds is None and turn % 6 == 0:
            try:
                winner, reason = await evaluate_debate(transcript, topic)
            except Exception:
                # If evaluation fails (e.g. no API key), break the loop gracefully
                winner, reason = "Unknown", "No evaluation available."
                break
            if winner in {name1, name2}:
                if console:
                    console.print(
                        Panel.fit(
                            f"Debate concluded early! {winner} has been declared the winner!\n\nReason: {reason}",
                            title="Winner",
                            style="bold magenta",
                        )
                    )
                return
            # If no clear winner, continue the debate
    # Time expired or loop ended
    # Final evaluation
    if console:
        console.print("\nEvaluating the debate...", style="yellow")
    try:
        winner, reason = await evaluate_debate(transcript, topic)
    except Exception as e:
        winner, reason = "Unknown", f"Evaluation failed: {e}"
    
    # Save debate log
    try:
        log_filepath = save_debate_log(
            topic=topic,
            name1=name1,
            name2=name2,
            backend1=backend1,
            backend2=backend2,
            model1=model1,
            model2=model2,
            transcript=transcript,
            winner=winner,
            reason=reason,
            time_limit=time_limit_seconds,
            allow_interruptions=allow_interruptions,
            research_snippet=research_snippet,
            current_events_context=current_events_context,
            debate_id=debate_id,
            persona1=persona1,
            persona2=persona2
        )
        
        # Create summary log
        summary_filepath = create_debate_summary_log()
        
        if console:
            console.print(f"📝 Debate log saved: {log_filepath}", style="green")
            console.print(f"📊 Summary updated: {summary_filepath}", style="green")
    except Exception as e:
        if console:
            console.print(f"⚠️  Failed to save debate log: {e}", style="yellow")
    
    if console:
        console.print(
            Panel.fit(
                f"{winner} is the winner of the debate!\n\nReason: {reason}",
                title="Final Result",
                style="bold magenta",
            )
        )


def choose_backend(console, backends: Dict[str, bool], agent_number: int) -> Tuple[str, str]:
    """Prompt the user to select a backend and model for an agent."""
    available = [name for name, available in backends.items() if available]
    if not available:
        console.print(
            "No LLM backends are configured.  Please set your API keys and/or install ollama.",
            style="red",
        )
        sys.exit(1)
    # Default to Ollama deepseek unless user wants to choose
    if backends.get("ollama"):
        use_default = Prompt.ask("Use default Ollama deepseek-r1:8b for agent {}?".format(agent_number), choices=["y", "n"], default="y")
        if use_default == "y":
            return "ollama", "deepseek-r1:8b"
    console.print(f"\nSelect a backend for {agent_number}:", style="bold")
    for idx, name in enumerate(available, 1):
        console.print(f"  {idx}. {name}")
    choice_idx = int(Prompt.ask("Enter number", choices=[str(i) for i in range(1, len(available) + 1)]))
    backend = available[choice_idx - 1]
    default_models = {
        "openai": ["gpt-4.1"],
        "gemini": ["gemini-1.5-pro"],
        "ollama": ["deepseek-r1:8b", "deepseek"],
    }
    models = default_models.get(backend, ["unknown"])
    console.print(f"Available models for {backend}: {', '.join(models)}")
    model = Prompt.ask("Enter model name", default=models[0])
    return backend, model


def choose_time_limit(console) -> Optional[int]:
    """Prompt the user to select a time limit or choose endless mode."""
    console.print("\nChoose debate mode:", style="bold")
    console.print("  1. Timed (60 seconds)")
    console.print("  2. Timed (5 minutes)")
    console.print("  3. Timed (10 minutes)")
    console.print("  4. Endless mode (philosophical battles)")
    mode = Prompt.ask("Enter option", choices=["1", "2", "3", "4"])
    if mode == "1":
        return 60
    if mode == "2":
        return 5 * 60
    if mode == "3":
        return 10 * 60
    return None



# === New Main Menu and Modes ===
def choose_main_mode(console) -> str:
    console.print("\n🧭 Select Application Mode", style="bold cyan")
    console.print("1. Business Task (agents solve a business problem)")
    console.print("2. Research (agents gather and summarize info)")
    console.print("3. War Room (4 agents collaborate to solve a challenge)")
    console.print("4. Debate (classic head-to-head debate)")
    mode = Prompt.ask("Choose mode", choices=["1", "2", "3", "4"], default="1")
    return mode

async def run_business_task(console, backends: Dict[str, bool]):
    console.print("\n💼 Business Task Mode", style="bold green")
    task = Prompt.ask("Describe your business problem or task")
    agent_roles = ["Analyst", "Strategist"]
    agent_names = [f"{role}Bot" for role in agent_roles]
    backend1, model1, backend2, model2 = auto_select_models(backends)
    personas = [await generate_agent_persona(agent_names[0], task), await generate_agent_persona(agent_names[1], task)]
    messages = [[{"role": "system", "content": f"You are {agent_names[i]}, a business agent. Persona: {personas[i]}"}] for i in range(2)]
    # Agents collaborate: Analyst proposes, Strategist refines
    user_prompt = f"Business Task: {task}. Analyst, propose a solution."
    messages[0].append({"role": "user", "content": user_prompt})
    reply1 = await get_model_reply(backend1, model1, messages[0])
    messages[0].append({"role": "assistant", "content": reply1})
    messages[1].append({"role": "user", "content": reply1})
    reply2 = await get_model_reply(backend2, model2, messages[1])
    messages[1].append({"role": "assistant", "content": reply2})
    console.print(Panel.fit(reply1, title=agent_names[0], style="bold blue"))
    console.print(Panel.fit(reply2, title=agent_names[1], style="bold green"))
    console.print("\n✅ Business solution generated.", style="bold yellow")

async def run_research_mode(console, backends: Dict[str, bool]):
    console.print("\n🔍 Research Mode", style="bold green")
    query = Prompt.ask("What would you like researched?")
    agent_roles = ["Researcher", "Summarizer"]
    agent_names = [f"{role}Bot" for role in agent_roles]
    backend1, model1, backend2, model2 = auto_select_models(backends)
    personas = [await generate_agent_persona(agent_names[0], query), await generate_agent_persona(agent_names[1], query)]
    messages = [[{"role": "system", "content": f"You are {agent_names[i]}, a research agent. Persona: {personas[i]}"}] for i in range(2)]
    # Researcher gathers info, Summarizer distills
    user_prompt = f"Research Task: {query}. Researcher, gather key information."
    messages[0].append({"role": "user", "content": user_prompt})
    reply1 = await get_model_reply(backend1, model1, messages[0])
    messages[0].append({"role": "assistant", "content": reply1})
    messages[1].append({"role": "user", "content": reply1})
    reply2 = await get_model_reply(backend2, model2, messages[1])
    messages[1].append({"role": "assistant", "content": reply2})
    console.print(Panel.fit(reply1, title=agent_names[0], style="bold blue"))
    console.print(Panel.fit(reply2, title=agent_names[1], style="bold green"))
    console.print("\n✅ Research summary generated.", style="bold yellow")

async def run_war_room(console, backends: Dict[str, bool]):
    console.print("\n⚔️ War Room Mode (4 Agents)", style="bold magenta")
    challenge = Prompt.ask("Describe the challenge or decision to solve")
    agent_roles = ["Analyst", "Strategist", "Skeptic", "Optimist"]
    agent_names = [f"{role}Bot" for role in agent_roles]
    backend1, model1, backend2, model2 = auto_select_models(backends)
    # Use two backends, alternate agents
    backends_models = [(backend1, model1), (backend2, model2), (backend1, model1), (backend2, model2)]
    personas = [await generate_agent_persona(agent_names[i], challenge) for i in range(4)]
    messages = [[{"role": "system", "content": f"You are {agent_names[i]}, a war room agent. Persona: {personas[i]}"}] for i in range(4)]
    # Initial prompt
    user_prompt = f"War Room Challenge: {challenge}. Analyst, start by analyzing the problem."
    messages[0].append({"role": "user", "content": user_prompt})
    reply0 = await get_model_reply(*backends_models[0], messages[0])
    messages[0].append({"role": "assistant", "content": reply0})
    # Strategist builds on Analyst
    messages[1].append({"role": "user", "content": reply0})
    reply1 = await get_model_reply(*backends_models[1], messages[1])
    messages[1].append({"role": "assistant", "content": reply1})
    # Skeptic critiques
    messages[2].append({"role": "user", "content": reply1})
    reply2 = await get_model_reply(*backends_models[2], messages[2])
    messages[2].append({"role": "assistant", "content": reply2})
    # Optimist proposes best-case solution
    messages[3].append({"role": "user", "content": reply2})
    reply3 = await get_model_reply(*backends_models[3], messages[3])
    messages[3].append({"role": "assistant", "content": reply3})
    # Display all agent outputs
    for i, reply in enumerate([reply0, reply1, reply2, reply3]):
        style = ["bold blue", "bold green", "bold red", "bold yellow"][i]
        console.print(Panel.fit(reply, title=agent_names[i], style=style))
    console.print("\n✅ War Room solution generated.", style="bold yellow")

async def main() -> None:
    console = Console() if Console else None
    if console:
        print_banner(console)
    backends = get_available_backends()
    if AsyncWebCrawler is not None:
        await display_current_headlines(console)
    mode = choose_main_mode(console)
    if mode == "1":
        await run_business_task(console, backends)
    elif mode == "2":
        await run_research_mode(console, backends)
    elif mode == "3":
        await run_war_room(console, backends)
    else:
        # === Debate Mode (legacy, now just a feature) ===
        name1, name2 = await generate_agent_names()
        console.print(f"\n🎭 Generated Agent Names: {name1} vs {name2}", style="bold green")
        backend1, model1, backend2, model2 = auto_select_models(backends)
        console.print(f"\n🤖 Auto-selected Models: {name1} ({backend1}/{model1}) vs {name2} ({backend2}/{model2})", style="bold blue")
        custom_topic, allow_interruptions = choose_debate_mode(console)
        if custom_topic:
            topics = [custom_topic]
            console.print(f"\n🎯 Custom Topic: {custom_topic}", style="bold green")
        else:
            topics = get_debate_topics()
        if console:
            if custom_topic:
                console.print("Using your custom topic for the debate.", style="green")
            else:
                table = Table(title="Debate Topics", header_style="bold magenta")
                table.add_column("Number", justify="center", style="cyan")
                table.add_column("Topic", style="green")
                for idx, topic in enumerate(topics, 1):
                    table.add_row(str(idx), topic)
                console.print(table)
        else:
            print("Debate Topics:")
            for idx, topic in enumerate(topics, 1):
                print(f"{idx}. {topic}")
        if custom_topic:
            topic = custom_topic
        else:
            topic_choice = int(Prompt.ask("Select a topic", choices=[str(i) for i in range(1, len(topics) + 1)]))
            topic = topics[topic_choice - 1]
        console.print(f"\n🎭 Generating personas for {name1} and {name2}...", style="bold cyan")
        persona1 = await generate_agent_persona(name1, topic)
        persona2 = await generate_agent_persona(name2, topic)
        if console:
            console.print(f"\n🎭 {name1}'s Persona:", style="bold blue")
            console.print(f"   Backstory: {persona1.get('backstory', 'N/A')}")
            console.print(f"   Personality: {persona1.get('personality', 'N/A')}")
            console.print(f"   Motivation: {persona1.get('debate_motivation', 'N/A')}")
            console.print(f"   Confidence: {persona1.get('confidence_level', 'N/A')}")
            console.print(f"\n🎭 {name2}'s Persona:", style="bold green")
            console.print(f"   Backstory: {persona2.get('backstory', 'N/A')}")
            console.print(f"   Personality: {persona2.get('personality', 'N/A')}")
            console.print(f"   Motivation: {persona2.get('debate_motivation', 'N/A')}")
            console.print(f"   Confidence: {persona2.get('confidence_level', 'N/A')}")
        time_limit = choose_time_limit(console)
        mode_info = "Interactive Mode" if allow_interruptions else "Standard Mode"
        research_info = "with Web Research" if AsyncWebCrawler is not None else "without Web Research"
        if console:
            console.print(
                Panel.fit(
                    f"Debate Topic: {topic}\n{name1} ({backend1}/{model1}) vs {name2} ({backend2}/{model2})\nTime limit: {'Endless' if time_limit is None else str(time_limit) + 's'}\nMode: {mode_info}\nResearch: {research_info}",
                    title="Battle Setup",
                    style="bold yellow",
                )
            )
        await conduct_debate(
            topic=topic,
            backend1=backend1,
            model1=model1,
            name1=name1,
            backend2=backend2,
            model2=model2,
            name2=name2,
            time_limit_seconds=time_limit,
            allow_interruptions=allow_interruptions,
        )


def create_debate_log_directory() -> str:
    """Create and return the path to the debate logs directory."""
    logs_dir = os.path.join(os.getcwd(), "debate_logs")
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def format_timestamp() -> str:
    """Format current timestamp for log files."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_debate_log(
    topic: str,
    name1: str,
    name2: str,
    backend1: str,
    backend2: str,
    model1: str,
    model2: str,
    transcript: List[Tuple[str, str]],
    winner: str,
    reason: str,
    time_limit: Optional[int],
    allow_interruptions: bool,
    research_snippet: Optional[str] = None,
    current_events_context: Optional[str] = None,
    debate_id: str = None,
    persona1: Optional[Dict[str, str]] = None,
    persona2: Optional[Dict[str, str]] = None
) -> str:
    """Save debate results to a markdown file.
    
    Returns the path to the saved log file.
    """
    logs_dir = create_debate_log_directory()
    
    if debate_id is None:
        debate_id = str(uuid.uuid4())[:8]
    
    timestamp = format_timestamp()
    filename = f"debate_{timestamp}_{debate_id}.md"
    filepath = os.path.join(logs_dir, filename)
    
    # Format the debate log
    log_content = f"""# 🏆 Debate Arena Results

## 📋 Debate Information
- **Date & Time:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Debate ID:** {debate_id}
- **Topic:** {topic}
- **Mode:** {'Interactive' if allow_interruptions else 'Standard'}
- **Time Limit:** {'Endless' if time_limit is None else f'{time_limit} seconds'}

## 🤖 Participants

### {name1}
- **Backend:** {backend1}
- **Model:** {model1}
"""
    
    # Add persona information if available
    if persona1:
        log_content += f"""
#### 🎭 Persona Details
- **Backstory:** {persona1.get('backstory', 'N/A')}
- **Personality:** {persona1.get('personality', 'N/A')}
- **Demographics:** {persona1.get('demographics', 'N/A')}
- **Likes:** {persona1.get('likes', 'N/A')}
- **Dislikes:** {persona1.get('dislikes', 'N/A')}
- **Debate Motivation:** {persona1.get('debate_motivation', 'N/A')}
- **Confidence Level:** {persona1.get('confidence_level', 'N/A')}
- **Debate Style:** {persona1.get('debate_style', 'N/A')}
- **Emotional State:** {persona1.get('emotional_state', 'N/A')}
- **Personal Stakes:** {persona1.get('personal_stakes', 'N/A')}
- **Communication Quirks:** {persona1.get('communication_quirks', 'N/A')}
"""
    
    log_content += f"""
### {name2}
- **Backend:** {backend2}
- **Model:** {model2}
"""
    
    # Add persona information if available
    if persona2:
        log_content += f"""
#### 🎭 Persona Details
- **Backstory:** {persona2.get('backstory', 'N/A')}
- **Personality:** {persona2.get('personality', 'N/A')}
- **Demographics:** {persona2.get('demographics', 'N/A')}
- **Likes:** {persona2.get('likes', 'N/A')}
- **Dislikes:** {persona2.get('dislikes', 'N/A')}
- **Debate Motivation:** {persona2.get('debate_motivation', 'N/A')}
- **Confidence Level:** {persona2.get('confidence_level', 'N/A')}
- **Debate Style:** {persona2.get('debate_style', 'N/A')}
- **Emotional State:** {persona2.get('emotional_state', 'N/A')}
- **Personal Stakes:** {persona2.get('personal_stakes', 'N/A')}
- **Communication Quirks:** {persona2.get('communication_quirks', 'N/A')}
"""
    
    log_content += f"""
## 🏆 Winner
**{winner}**

### 🎯 Judge's Reasoning
{reason}

## 📝 Full Transcript

"""
    
    # Add transcript
    for i, (speaker, message) in enumerate(transcript, 1):
        log_content += f"### Turn {i}: {speaker}\n\n{message}\n\n---\n\n"
    
    # Add research context if available
    if research_snippet or current_events_context:
        log_content += "## 🔍 Research & Context\n\n"
        if research_snippet:
            log_content += f"### Topic Research\n\n{research_snippet}\n\n"
        if current_events_context:
            log_content += f"### Current Events Context\n\n{current_events_context}\n\n"
    
    # Add debate statistics
    total_turns = len(transcript)
    name1_turns = sum(1 for speaker, _ in transcript if speaker == name1)
    name2_turns = sum(1 for speaker, _ in transcript if speaker == name2)
    
    log_content += f"""## 📊 Debate Statistics
- **Total Turns:** {total_turns}
- **{name1} Turns:** {name1_turns}
- **{name2} Turns:** {name2_turns}
- **Average Turn Length:** {sum(len(msg) for _, msg in transcript) // total_turns if total_turns > 0 else 0} characters

## 🎯 Analysis
This debate was conducted using the SynthAgent Arena platform with AI-powered participants who were given detailed personas to create more realistic and engaging debates. The transcript shows the full exchange of arguments, and the winner was determined by an AI judge evaluating clarity, evidence, logic, and rhetorical skill.

---
*Generated by SynthAgent Arena - AI Debate Platform*
"""
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    return filepath


def create_debate_summary_log() -> str:
    """Create a summary log of all debates."""
    logs_dir = create_debate_log_directory()
    summary_file = os.path.join(logs_dir, "debate_summary.md")
    
    # Get all debate log files
    log_files = [f for f in os.listdir(logs_dir) if f.startswith("debate_") and f.endswith(".md")]
    log_files.sort(reverse=True)  # Most recent first
    
    summary_content = """# 📊 SynthAgent Arena - Debate Summary

## 🏆 Recent Debates

"""
    
    for log_file in log_files[:10]:  # Show last 10 debates
        try:
            with open(os.path.join(logs_dir, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract key information
                lines = content.split('\n')
                topic = ""
                winner = ""
                timestamp = ""
                
                for line in lines:
                    if line.startswith("- **Topic:**"):
                        topic = line.replace("- **Topic:**", "").strip()
                    elif line.startswith("**") and "Winner" in line:
                        winner = line.replace("**", "").replace("**", "").strip()
                    elif line.startswith("- **Date & Time:**"):
                        timestamp = line.replace("- **Date & Time:**", "").strip()
                        break
                
                if topic and winner and timestamp:
                    summary_content += f"### {timestamp}\n"
                    summary_content += f"- **Topic:** {topic}\n"
                    summary_content += f"- **Winner:** {winner}\n"
                    summary_content += f"- **Log:** [{log_file}]({log_file})\n\n"
                    
        except Exception:
            continue
    
    summary_content += f"""
## 📈 Statistics
- **Total Debates Logged:** {len(log_files)}
- **Latest Debate:** {log_files[0] if log_files else 'None'}

---
*Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_content)
    
    return summary_file


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDebate terminated by user.")