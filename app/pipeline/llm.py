"""
LLM Service Module: Intent Detection + Conversational Response Generation
Streams tokens in real time and classifies user intent in a single pass for minimum latency.
"""

import re
import logging
from typing import AsyncGenerator, Dict, List, Optional, Tuple
from app.config import settings

logger = logging.getLogger("voicebot.llm")

SYSTEM_PROMPT = """You are an intelligent, friendly, and low-latency real-time voice assistant.
Your responses will be spoken aloud to the user.

RULES:
1. Always start your response with an intent tag:
   [INTENT: <Category>]
   Choose <Category> from: [Greeting, Question, Command, ChitChat, Clarification, Feedback, Farewell, General]
2. Immediately follow the intent tag with your conversational voice response.
3. Provide a direct, complete, and engaging spoken answer (typically 1 to 3 full sentences). Ensure you completely answer the user's inquiry without trailing off.
4. Do NOT use markdown symbols, asterisks, bullet points, numbered lists, URLs, or tables since they cannot be spoken naturally.
5. Example:
   [INTENT: Greeting] Hello! How can I help you today?
   [INTENT: Question] The capital of France is Paris, famous for its art, culture, and iconic landmarks like the Eiffel Tower.
"""


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.model = settings.LLM_MODEL
        self._groq_client = None
        self._openai_client = None

    def _get_groq_client(self):
        if self._groq_client is None:
            from groq import AsyncGroq
            self._groq_client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import AsyncOpenAI
            self._openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return self._openai_client

    async def stream_response(
        self,
        messages_history: List[Dict[str, str]],
        user_message: str
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Yields (chunk_text, detected_intent).
        detected_intent is extracted as early as possible (usually in first chunk).
        chunk_text contains only the clean conversational text stripped of [INTENT: ...].
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(messages_history)
        messages.append({"role": "user", "content": user_message})

        if self.provider == "groq":
            async for chunk in self._stream_groq(messages):
                yield chunk
        elif self.provider == "openai":
            async for chunk in self._stream_openai(messages):
                yield chunk
        elif self.provider == "mock":
            async for chunk in self._stream_mock(user_message):
                yield chunk
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def _stream_groq(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        client = self._get_groq_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_tokens=450,
                stream=True
            )
            async for item in self._process_stream(stream):
                yield item
        except Exception as e:
            logger.error(f"Groq LLM streaming error: {e}", exc_info=True)
            raise

    async def _stream_openai(
        self,
        messages: List[Dict[str, str]]
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        client = self._get_openai_client()
        try:
            stream = await client.chat.completions.create(
                model=self.model or "gpt-4o-mini",
                messages=messages,
                temperature=0.6,
                max_tokens=250,
                stream=True
            )
            async for item in self._process_stream(stream):
                yield item
        except Exception as e:
            logger.error(f"OpenAI LLM streaming error: {e}", exc_info=True)
            raise

    async def _stream_mock(
        self,
        user_message: str
    ) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        intent = "Greeting" if "hello" in user_message.lower() else "Question"
        yield ("", intent)
        sample_response = f"I received your message: '{user_message}'. How can I assist you further?"
        for word in sample_response.split(" "):
            yield (word + " ", None)

    async def _process_stream(self, stream) -> AsyncGenerator[Tuple[str, Optional[str]], None]:
        """
        Parses stream to extract [INTENT: ...] in the first few tokens,
        then streams clean speech text as it arrives.
        """
        buffer = ""
        intent_detected: Optional[str] = None
        intent_extracted = False
        pattern = re.compile(r"\[INTENT:\s*([a-zA-Z0-9_\-\s]+)\]", re.IGNORECASE)

        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            if not token:
                continue

            if not intent_extracted:
                buffer += token
                match = pattern.search(buffer)
                if match:
                    intent_detected = match.group(1).strip()
                    intent_extracted = True
                    # Remove the [INTENT: ...] tag from remaining buffer
                    remainder = buffer[match.end():].lstrip()
                    buffer = ""
                    if remainder:
                        yield (remainder, intent_detected)
                    else:
                        yield ("", intent_detected)
                elif len(buffer) > 40:
                    # Fallback if model did not follow tag format within 40 characters
                    intent_detected = "General"
                    intent_extracted = True
                    yield (buffer, intent_detected)
                    buffer = ""
            else:
                yield (token, None)

        if buffer:
            yield (buffer, intent_detected or "General")


llm_service = LLMService()
