# Changelog

## [2024-01-XX] - OpenAI Responses API Update

### Changed
- **BREAKING**: Updated OpenAI integration to use the new Responses API instead of the deprecated ChatCompletion API
- Updated all OpenAI API calls to use the new `/v1/responses` endpoint
- Changed default model from `gpt-4o` to `gpt-4.1` (the new recommended model)
- Updated backend availability check to use `requests` library instead of `openai` library
- Removed fallback logic for deprecated models (`gpt-4o`, `gpt-3.5-turbo`)

### Technical Changes
- Added `requests` library import for HTTP API calls
- Updated `call_openai()` function to use the new API format with `input`, `text`, `reasoning`, and `tools` fields
- Updated `generate_topics_via_openai()` function to use the new API
- Updated `evaluate_debate()` function to use the new API
- Updated `get_available_backends()` to check for `requests` library instead of `openai` library
- Updated default model suggestions to only include `gpt-4.1`

### API Format Changes
The new API uses this format:
```json
{
  "model": "gpt-4.1",
  "input": [{"role": "user", "content": "..."}],
  "text": {"format": {"type": "text"}},
  "reasoning": {},
  "tools": [],
  "temperature": 0.7,
  "max_output_tokens": 512,
  "top_p": 1,
  "store": true
}
```

### Requirements
- `requests` library must be installed: `pip install requests`
- `OPENAI_API_KEY` environment variable must be set
- Uses `gpt-4.1` model (no longer supports deprecated models) 