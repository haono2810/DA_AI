"""
=============================================================================
  DAMON AI - DỊCH VỤ PHÁT ÂM GIỌNG NÓI NEURAL TTS (TEXT-TO-SPEECH)
=============================================================================
Tạo âm thanh giọng đọc tự nhiên (Microsoft Edge Neural Voice) đa ngôn ngữ:
  Tiếng Việt, Tiếng Anh, Tiếng Trung, Tiếng Hàn, Tiếng Nga.
"""

import asyncio

VOICE_MAP = {
    "vi": ("vi-VN-NamMinhNeural", "vi-VN-HoaiMyNeural"),
    "en": ("en-US-GuyNeural", "en-US-JennyNeural"),
    "zh": ("zh-CN-YunxiNeural", "zh-CN-XiaoxiaoNeural"),
    "ko": ("ko-KR-SunHiNeural", "ko-KR-InJoonNeural"),
    "ru": ("ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural"),
}


def _clean_text(text):
    """Làm sạch văn bản: lấy từ đầu tiên trước dấu '/' hoặc ',' để đọc tự nhiên."""
    if not text:
        return ""
    return text.replace(" / ", "/").split("/")[0].split(",")[0].strip()


async def _synthesize_one(edge_tts_mod, text, lang):
    """Phát âm một đoạn văn bản, tự động fallback sang giọng dự phòng nếu lỗi."""
    primary, backup = VOICE_MAP.get(lang, VOICE_MAP["vi"])
    for voice in (primary, backup):
        try:
            chunks = []
            comm = edge_tts_mod.Communicate(text, voice, rate="-5%")
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    chunks.append(chunk["data"])
            if chunks:
                return b"".join(chunks)
        except Exception as e:
            print(f"⚠️ Lỗi giọng {voice} ({lang}): {e}")
    return b""


async def _synthesize_edge_tts(text="", lang="vi", secondary_text="", secondary_lang="en"):
    import edge_tts

    audio_parts = []
    for txt, lng in [(text, lang), (secondary_text, secondary_lang)]:
        cleaned = _clean_text(txt)
        if cleaned:
            audio_parts.append(await _synthesize_one(edge_tts, cleaned, lng))

    return b"".join(audio_parts)


def generate_neural_tts(text="", lang="vi", secondary_text="", secondary_lang="en", en_text="", vi_text=""):
    """
    Sinh luồng byte âm thanh MP3 từ văn bản đa ngôn ngữ.
    Hỗ trợ tương thích ngược với tham số cũ (en_text, vi_text).
    """
    # Xử lý tương thích ngược
    if not text and (vi_text or en_text):
        if vi_text:
            text, lang = vi_text, "vi"
            if en_text:
                secondary_text, secondary_lang = en_text, "en"
        else:
            text, lang = en_text, "en"

    if not text and not secondary_text:
        return b""

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            _synthesize_edge_tts(text=text, lang=lang, secondary_text=secondary_text, secondary_lang=secondary_lang)
        )
    finally:
        loop.close()
