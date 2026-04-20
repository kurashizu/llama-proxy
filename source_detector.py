import json
import re
from typing import Dict, List, Optional


class SourceDetector:
    """
    Hermes Llama Proxy - Source Detection Engine (Strict Version)
    采用高精度指纹特征组匹配，确保来源识别的准确性。
    """

    # 精准匹配规则定义 (基于 System Prompt 中的排他性指令段落)
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
        """从各种格式的消息内容中提取纯文本"""
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
        高精度识别逻辑：
        1. 针对 System Prompt 执行特征组严格匹配
        2. 针对 Body 执行 OpenAI 标准字段匹配
        3. 针对 User-Agent 执行客户端指纹匹配
        """

        # --- 1. System Prompt 深度指纹扫描 ---
        system_msg = next(
            (
                cls._extract_text(m.get("content", ""))
                for m in messages
                if m.get("role") == "system"
            ),
            "",
        )

        # 扫描 System Prompt 以及前几条消息 (针对 TitleGen 指令可能出现在 User 消息中的情况)
        search_texts = [system_msg] if system_msg else []
        if len(messages) <= 3:
            for m in messages:
                content = cls._extract_text(m.get("content", ""))
                if content and content not in search_texts:
                    search_texts.append(content)

        for text in search_texts:
            # 1.1 尝试匹配 Discord 专用格式
            for p in cls.PATTERNS["Discord"]:
                if re.search(p, text):
                    return "Discord"

            # 1.2 尝试匹配 CLI 专用指令
            for p in cls.PATTERNS["CLI"]:
                if re.search(p, text):
                    return "CLI"

            # 1.3 尝试匹配 标题生成 任务
            for p in cls.PATTERNS["TitleGen"]:
                if re.search(p, text):
                    return "TitleGen"

            # 1.4 通用 platform 注入正则 (匹配完整 tag)
            m_plt = re.search(
                r"\bplatform:\s*([A-Za-z0-9_]+)\b", system_msg, re.IGNORECASE
            )
            if m_plt:
                return m_plt.group(1).upper()

        # --- 2. JSON Body 显式字段检测 ---
        # 很多 WebUI 或自定义 Agent 会在 JSON 顶层传 user 字段
        body_user = body.get("user", "")
        if body_user and isinstance(body_user, str):
            # 过滤掉默认的 generic 占位符
            if body_user.lower() not in ["user", "default", "none"]:
                return body_user

        # --- 3. User-Agent 客户端指纹识别 ---
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

            # 保底：取 UA 第一个有意义的部分
            ua_parts = ua.split("/")
            if ua_parts:
                clean_ua = ua_parts[0].strip()
                if len(clean_ua) < 15:  # 防止长字符串刷屏
                    return clean_ua

        return "Unknown"


# 导出函数
detect_source = SourceDetector.detect
