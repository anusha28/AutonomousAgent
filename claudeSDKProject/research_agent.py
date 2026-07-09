"""
Protein Digestion Research Agent
Demonstrates multi-turn conversational research using ClaudeSDKClient.

Examples:
  1. deep_dive_conversation  — a single research thread that builds context
                               across multiple follow-up questions.
  2. comparative_analysis    — two sequential topics analyzed in the same
                               session so Claude can draw cross-topic comparisons.
  3. session_memory_demo     — shows how to list past sessions, retrieve their
                               messages, and resume a previous conversation.
  4. tool_augmented_research — uses web_search and download_pdfs tools so Claude
                               can find and read real papers during the conversation.
"""

import asyncio
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    ResultMessage,
    list_sessions,
    get_session_messages,
)
from tools import research_tools_server

SYSTEM_PROMPT = """You are a health research assistant specializing in human physiology and nutrition science.
Provide clear, evidence-based answers to questions about how the body works, digestion, and nutrition.
Be concise but thorough, citing key biological mechanisms."""


# ── helpers ──────────────────────────────────────────────────────────────────

def print_divider(label: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


async def collect_response(client: ClaudeSDKClient) -> str:
    """Iterate receive_response(), print text and surface tool calls/results."""
    full_text = ""
    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                    full_text += block.text
                elif isinstance(block, ToolUseBlock):
                    print(f"\n[tool call → {block.name}]")
        elif isinstance(message, ToolResultBlock):
            print(f"[tool result received]")
        elif isinstance(message, ResultMessage) and message.subtype != "success":
            print(f"[result subtype={message.subtype}]")
    return full_text


# ── example 1: deep-dive conversation ────────────────────────────────────────

async def deep_dive_conversation() -> None:
    """
    A single research session that builds understanding step by step.
    Each follow-up references what Claude already explained — demonstrating
    that the session retains the full conversation history.
    """
    print_divider("EXAMPLE 1 — Deep-Dive Conversation")
    print("Topic: protein digestion, explored across 4 connected turns.\n")

    options = ClaudeAgentOptions(system_prompt=SYSTEM_PROMPT)

    async with ClaudeSDKClient(options=options) as client:
        # Turn 1 — establish the foundation
        print(">> Turn 1: gastric phase")
        await client.query(
            "What happens to protein when it enters the stomach? "
            "Explain the roles of pepsin and hydrochloric acid."
        )
        await collect_response(client)

        # Turn 2 — advance to the next organ, building on turn 1
        print("\n>> Turn 2: small intestine (follow-up)")
        await client.query(
            "You mentioned pepsin breaks protein into polypeptides. "
            "What enzymes continue that work in the small intestine, "
            "and how are the resulting amino acids absorbed?"
        )
        await collect_response(client)

        # Turn 3 — zoom in on a mechanism introduced in turn 2
        print("\n>> Turn 3: PepT1 transporter (drill-down)")
        await client.query(
            "You mentioned the PepT1 transporter for di/tripeptides. "
            "How does it differ mechanistically from the Na⁺-dependent "
            "amino acid cotransporters?"
        )
        await collect_response(client)

        # Turn 4 — synthesise everything into a practical recommendation
        print("\n>> Turn 4: synthesis — practical nutrition takeaway")
        await client.query(
            "Given everything we've covered about digestion and absorption, "
            "what practical advice would you give someone who wants to "
            "maximise muscle-protein synthesis after a workout?"
        )
        await collect_response(client)


# ── example 2: comparative analysis in one session ───────────────────────────

async def comparative_analysis() -> None:
    """
    Analyse two related topics in the same session so Claude can later
    draw explicit comparisons — something impossible with stateless query().
    """
    print_divider("EXAMPLE 2 — Comparative Analysis (Same Session)")
    print("Topics: whey vs. casein digestion kinetics, then a direct comparison.\n")

    options = ClaudeAgentOptions(system_prompt=SYSTEM_PROMPT)

    async with ClaudeSDKClient(options=options) as client:
        # Topic A
        print(">> Topic A: whey protein")
        await client.query(
            "Describe how whey protein is digested and absorbed — "
            "focus on speed, amino acid profile, and peak blood levels."
        )
        await collect_response(client)

        # Topic B
        print("\n>> Topic B: casein protein")
        await client.query(
            "Now do the same for casein protein — digestion speed, "
            "amino acid delivery pattern, and any unique gastric behaviour."
        )
        await collect_response(client)

        # Cross-topic comparison — only possible because both topics are in memory
        print("\n>> Cross-topic comparison")
        await client.query(
            "Based on what you've told me about both whey and casein, "
            "when is each protein type most beneficial during the day "
            "and why? Cite the mechanistic differences you described."
        )
        await collect_response(client)

        # Deepen — push for nuance the model hasn't yet volunteered
        print("\n>> Nuance: leucine threshold hypothesis")
        await client.query(
            "Does the leucine threshold hypothesis change which protein "
            "you'd recommend post-workout? Relate it back to the absorption "
            "kinetics you described for whey and casein."
        )
        await collect_response(client)


# ── example 3: session memory — list, inspect, resume ────────────────────────

async def session_memory_demo() -> None:
    """
    Shows the session-management API:
      • Start a session and record its ID.
      • List past sessions.
      • Retrieve that session's messages.
      • Resume the session with a follow-up question.
    """
    print_divider("EXAMPLE 3 — Session Memory & Resume")

    options = ClaudeAgentOptions(system_prompt=SYSTEM_PROMPT)
    captured_session_id: str | None = None

    # ── phase A: create a new session ──
    print(">> Phase A: starting a new research session\n")
    async with ClaudeSDKClient(options=options) as client:
        await client.query(
            "What is the role of insulin in regulating amino acid uptake "
            "into muscle cells after a protein-rich meal?"
        )
        await collect_response(client)

        await client.query(
            "How does mTOR fit into that signalling pathway?"
        )
        await collect_response(client)

    # ── phase B: inspect past sessions ──
    # Grab the most-recently-modified session (the one we just created)
    print("\n>> Phase B: listing recent sessions")
    sessions = list_sessions(limit=5)
    for s in sessions:
        tag_label = f"  [tag: {s.tag}]" if s.tag else ""
        print(f"  • {s.session_id[:8]}…  {s.summary or s.first_prompt or '(no summary)'}{tag_label}")

    # Use the most recent session for phases C and D
    if sessions:
        captured_session_id = sessions[0].session_id
        print(f"\n[using session: {captured_session_id}]")

    # ── phase C: retrieve messages from our session ──
    if captured_session_id:
        print(f"\n>> Phase C: retrieving messages from session {captured_session_id[:8]}…")
        messages = get_session_messages(captured_session_id)
        print(f"  {len(messages)} message(s) found.")
        for msg in messages:
            role_label = msg.type.upper()
            # Show a short preview of the message content
            raw = msg.message
            content = raw.get("content", "")
            if isinstance(content, list):
                preview = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            else:
                preview = str(content)
            print(f"  [{role_label}] {preview[:120].strip()}{'…' if len(preview) > 120 else ''}")

    # ── phase D: resume the session ──
    if captured_session_id:
        print(f"\n>> Phase D: resuming session {captured_session_id[:8]}… with a follow-up")
        resume_options = ClaudeAgentOptions(
            system_prompt=SYSTEM_PROMPT,
            resume=captured_session_id,
        )
        async with ClaudeSDKClient(options=resume_options) as client:
            await client.query(
                "Connecting what you said about insulin and mTOR: "
                "does leucine directly activate mTOR, or does it work "
                "indirectly through insulin signalling?"
            )
            await collect_response(client)


# ── example 4: tool-augmented research ───────────────────────────────────────

async def tool_augmented_research() -> None:
    """
    Equips Claude with web_search and download_pdfs tools so it can:
      1. Search for real, recent papers on a topic.
      2. Download any PDF links it finds to ./Research Papers/.
      3. Summarise and critique the literature in a follow-up turn,
         drawing on the extracted PDF text it just retrieved.

    The conversation is multi-turn: Claude searches first, then reflects
    on what it found, and finally synthesises a research summary — all
    within one session so context accumulates across turns.
    """
    print_divider("EXAMPLE 4 — Tool-Augmented Research")
    print("Claude will search the web, download PDFs, and synthesise findings.\n")

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"research_tools": research_tools_server},
        allowed_tools=[
            "mcp__research_tools__web_search",
            "mcp__research_tools__download_pdfs",
        ],
    )

    async with ClaudeSDKClient(options=options) as client:
        # Turn 1 — search then download every resolved PDF URL
        print(">> Turn 1: search for papers and download PDFs")
        await client.query(
            "Use the web_search tool to find the 10 most relevant recent research "
            "papers on 'protein digestion amino acid absorption mechanisms small intestine PepT1'. "
            "The tool returns a pdf_urls list where each article URL has been resolved "
            "to its direct PDF download link. Pass ALL of those pdf_urls to the "
            "download_pdfs tool so every paper gets saved locally."
        )
        await collect_response(client)

        # Turn 2 — ask Claude to reflect on what it found
        print("\n>> Turn 2: summarise what was found")
        await client.query(
            "Based on the search results and any PDFs you downloaded, "
            "give me a structured summary of the key findings: "
            "which sources look most credible, what mechanisms they cover, "
            "and what gaps or controversies exist in the literature?"
        )
        await collect_response(client)

        # Turn 3 — push for a specific synthesis using the downloaded content
        print("\n>> Turn 3: deep synthesis")
        await client.query(
            "From the papers you retrieved, what is the current scientific consensus "
            "on the relative importance of PepT1 di/tripeptide transport vs. "
            "free amino acid transporters for overall protein absorption efficiency? "
            "Cite specific sources where possible."
        )
        await collect_response(client)


# ── entry point ───────────────────────────────────────────────────────────────

async def main() -> None:
    print("Protein Digestion Research Agent")
    print("Multi-turn conversational mode — session context is preserved\n")

    await deep_dive_conversation()
    await comparative_analysis()
    await session_memory_demo()
    await tool_augmented_research()


if __name__ == "__main__":
    asyncio.run(main())
