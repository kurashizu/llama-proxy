.hermes/llama-proxy/source_detector.py#L1-200
import json
import re
from typing import Dict, List, Optional


class SourceDetector:
    """
    Hermes Llama Proxy - Source Detection Engine (Strict Version)
    High-precision fingerprint/group matching for reliable source detection.
    """

    # Precise matching rules (based on exclusive instruction blocks commonly found in System Prompts)
    PATTERNS = {
        "CLI": [
            r"You are a CLI AI Agent",
            r"simple text renderable inside a terminal",
            r"Try not to use markdown but simple text",
        ],
        "Discord": [
            r"Source:\s*Discord\s*\(group:",
            r"Source:\s*Discord\s*\(#",
            r"Source:\s*Discord\s*\(via",
        ],
        "TitleGen": [
            r"Generate a short, descriptive title",
            r"Return ONLY the title text",
            r"No quotes, no punctuation at the end, no prefixes",
            r"generates concise titles",
            r"Generate a short title",
            r"Respond only with the title text",
            r"title for this chat",
            r"summarize the conversation",
            r"short name for this conversation",
        ],
    }

    @staticmethod
    def _extract_text(content) -> str:
        """
        Extract plain text from various message content formats.

        - If content is a string, return it directly.
        - If content is a list (multimodal), join 'text' parts from dict elements.
        - Otherwise, coerce to string.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        return str(content)

    @classmethod
    def detect(cls, messages: List[Dict], body: Dict, headers: Dict) -> str:
        """
        High-precision detection logic:
        1. Perform fingerprint matching against the System Prompt and nearby messages.
        2. Inspect top-level JSON 'user' field in the body (many WebUIs or custom agents set it).
        3. Use the User-Agent header for client fingerprinting.

        Returns a short identifier like "Discord", "CLI", "TitleGen", "Postman", "Python-Req", etc.
        Falls back to "Unknown" if no reliable match is found.
        """

        # --- 1. Deep scan the System Prompt and the first few messages ---
        system_msg = next(
            (
                cls._extract_text(m.get("content", ""))
                for m in messages
                if m.get("role") == "system"
            ),
            "",
        )

        # Collect candidate texts to scan (system prompt + up to first 3 messages).
        search_texts = [system_msg] if system_msg else []
        if len(messages) <= 3:
            for m in messages:
                content = cls._extract_text(m.get("content", ""))
                if content and content not in search_texts:
                    search_texts.append(content)

        for text in search_texts:
            # 1.1 Try Discord-specific patterns
            for p in cls.PATTERNS["Discord"]:
                if re.search(p, text):
                    return "Discord"

            # 1.2 Try CLI-specific patterns
            for p in cls.PATTERNS["CLI"]:
                if re.search(p, text):
                    return "CLI"

            # 1.3 Try Title generation task patterns
            for p in cls.PATTERNS["TitleGen"]:
                if re.search(p, text):
                    return "TitleGen"

            # 1.4 Generic platform injection regex (match a full tag like `platform: NAME`)
            m_plt = re.search(
                r"\bplatform:\s*([A-Za-z0-9_]+)\b", system_msg, re.IGNORECASE
            )
            if m_plt:
                return m_plt.group(1).upper()

        # --- 2. Explicit JSON body field detection ---
        # Many WebUIs or custom agents include a top-level "user" field in the JSON body.
        body_user = body.get("user", "")
        if body_user and isinstance(body_user, str):
            # Filter out common generic placeholders
            if body_user.lower() not in ["user", "default", "none"]:
                return body_user

        # --- 3. User-Agent client fingerprinting ---
        ua = headers.get("User-Agent", "")
        if ua:
            ua_lower = ua.lower()
            if "discord" in ua_lower:
                return "Discord"
            if "openwebui" in ua_lower or "open-webui" in ua_lower:
                return "WebUI"
            if "python-requests" in ua_lower:
                return "Python-Req"
            if "aiohttp" in ua_lower:
                return "AioHttp"
            if "curl" in ua_lower:
                return "Curl"
            if "postman" in ua_lower:
                return "Postman"

            # Fallback: return the first meaningful User-Agent token if it's reasonably short
            ua_parts = ua.split("/")
            if ua_parts:
                clean_ua = ua_parts[0].strip()
                # Avoid returning overly long strings as the source identifier
                if len(clean_ua) < 15:
                    return clean_ua

        return "Unknown"


# Exported helper
detect_source = SourceDetector.detect
