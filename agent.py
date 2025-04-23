from __future__ import annotations

import logging
from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.plugins import openai
from pathlib import Path

load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("my-worker")
logger.setLevel(logging.INFO)


instr_files = list(Path(__file__).parent.rglob("instructions.txt"))
if not instr_files:
    logger.error("No instruction.txt found anywhere under the project root")
else:
    kb_path = instr_files[0]
    instructions_text = kb_path.read_text(encoding="utf-8")
    logger.error(f"Loaded instructions from {kb_path!r}")

async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    participant = await ctx.wait_for_participant()

    run_multimodal_agent(ctx, participant)

    logger.info("agent started")


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
    instructions=instructions_text,
    modalities=["audio", "text"],
    voice="alloy",
    )

    # create a chat context with chat history, these will be synchronized with the server
    # upon session establishment
    chat_ctx = llm.ChatContext()
    chat_ctx.append(
        text="""ابدأ المحادثة باللهجة السعودية بحفاوة:  
أهلاً هلا، أنا فاطمة من خدمة مدن أرامكو الذكيّة. كيف أقدر أخدمك اليوم؟  
""",
        role="assistant",
    )

    agent = MultimodalAgent(
        model=model,
        chat_ctx=chat_ctx
    )
    # agent.enable_audio_output()
    agent.start(ctx.room, participant)

    # to enable the agent to speak first
    agent.generate_reply()


def main():
    """Start the Aramco Smart Cities agent."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound",
        )
    )

if __name__ == "__main__":
    main()