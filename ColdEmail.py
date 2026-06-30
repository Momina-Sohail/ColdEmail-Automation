import os

import litellm
import streamlit as st
from crewai import Agent, Crew, LLM, Task
from dotenv import load_dotenv

load_dotenv()

# Remove cache_breakpoint from Groq calls (same fix as main.py)
_original_completion = litellm.completion


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        for msg in kwargs["messages"]:
            if isinstance(msg, dict) and "cache_breakpoint" in msg:
                del msg["cache_breakpoint"]
    return _original_completion(*args, **kwargs)


litellm.completion = _patched_completion


def get_llm(api_key: str) -> LLM:
    return LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=api_key,
        max_tokens=4096,
    )


def build_cold_email_crew(llm: LLM, context: dict) -> Crew:
    writer = Agent(
        role="Cold Email Outreach Specialist",
        goal=(
            "Write concise, personalized cold emails that get replies "
            "without sounding spammy or generic."
        ),
        backstory=(
            "You have 10+ years in B2B sales and outreach. You write emails "
            "that are short, specific, and focused on the recipient's problems."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Write a cold outreach email with these details:

**Your company:** {context["your_company"]}
**Your name & role:** {context["your_name"]} — {context["your_role"]}
**What you offer:** {context["your_offer"]}

**Target company:** {context["target_company"]}
**Target contact:** {context["target_contact"] or "Hiring manager / relevant lead"}
**Contact role:** {context["target_role"] or "Not specified"}

**Extra context from user:**
{context["extra_context"] or "None"}

Requirements:
- Subject line on the first line as "Subject: ..."
- Professional but warm tone
- Under 150 words in the body
- One clear call to action
- Mention something specific about the target company when possible
""",
        expected_output=(
            "A complete cold email with subject line, greeting, body, "
            "and sign-off ready to send."
        ),
        agent=writer,
    )

    return Crew(agents=[writer], tasks=[task], llm=llm, verbose=False)


def build_reply_crew(llm: LLM, context: dict) -> Crew:
    replier = Agent(
        role="Professional Email Reply Specialist",
        goal=(
            "Draft thoughtful replies to company emails that move the "
            "conversation forward and reflect the sender's goals."
        ),
        backstory=(
            "You are an expert communicator who handles business correspondence. "
            "You match tone, answer questions directly, and keep emails brief."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Draft a reply to the incoming email below.

**Your company:** {context["your_company"]}
**Your name & role:** {context["your_name"]} — {context["your_role"]}
**What you offer:** {context["your_offer"]}

**Company you are talking to:** {context["target_company"]}

**Incoming email to reply to:**
---
{context["incoming_email"]}
---

**Instructions for this reply:**
{context["reply_instructions"] or "Respond professionally and advance the conversation toward a meeting or next step."}

Requirements:
- Do NOT include "Subject:" unless a new subject is clearly needed
- Address every question or point in the incoming email
- Keep the reply under 200 words unless the incoming email requires more detail
- End with a clear next step
- Sign off with the sender's name
""",
        expected_output=(
            "A polished email reply ready to copy and send, with appropriate "
            "greeting and sign-off."
        ),
        agent=replier,
    )

    return Crew(agents=[replier], tasks=[task], llm=llm, verbose=False)


def sidebar_config() -> dict | None:
    st.sidebar.header("Configuration")

    api_key = st.sidebar.text_input(
        "Groq API Key",
        value=st.session_state.get("api_key", os.getenv("GROQ_API_KEY", "")),
        type="password",
        placeholder="Paste your Groq API key here",
        help="Updates in real time — used immediately on the next generate action.",
        key="api_key_input",
    )
    st.session_state["api_key"] = api_key

    st.sidebar.divider()
    st.sidebar.subheader("Your Company")

    your_company = st.sidebar.text_input(
        "Company name",
        placeholder="e.g. Acme Solutions",
        key="your_company",
    )
    your_name = st.sidebar.text_input(
        "Your name",
        placeholder="e.g. Jane Doe",
        key="your_name",
    )
    your_role = st.sidebar.text_input(
        "Your role",
        placeholder="e.g. Founder / Sales Lead",
        key="your_role",
    )
    your_offer = st.sidebar.text_area(
        "What you offer",
        placeholder="Brief value prop — what problem you solve and for whom",
        height=100,
        key="your_offer",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Target Company")

    target_company = st.sidebar.text_input(
        "Company to email",
        placeholder="e.g. Stripe, Google, local startup",
        key="target_company",
    )
    target_contact = st.sidebar.text_input(
        "Contact name (optional)",
        placeholder="e.g. Alex Chen",
        key="target_contact",
    )
    target_role = st.sidebar.text_input(
        "Contact role (optional)",
        placeholder="e.g. Head of Partnerships",
        key="target_role",
    )

    if not api_key.strip():
        st.sidebar.warning("Add your Groq API key to generate emails.")
        return None

    return {
        "api_key": api_key.strip(),
        "your_company": your_company.strip() or "Your Company",
        "your_name": your_name.strip() or "Your Name",
        "your_role": your_role.strip() or "Your Role",
        "your_offer": your_offer.strip() or "Our product or service",
        "target_company": target_company.strip() or "the company",
        "target_contact": target_contact.strip(),
        "target_role": target_role.strip(),
    }


def main():
    st.set_page_config(
        page_title="Cold Email Replier",
        page_icon="✉️",
        layout="wide",
    )

    st.title("✉️ Cold Email Replier")
    st.caption(
        "Generate personalized cold outreach emails and automated replies "
        "powered by CrewAI + Groq."
    )

    config = sidebar_config()
    if config is None:
        st.info("Configure your API key and company details in the sidebar to get started.")
        return

    tab_generate, tab_reply = st.tabs(["Generate Cold Email", "Reply to Email"])

    with tab_generate:
        st.subheader("Generate a cold email")
        extra_context = st.text_area(
            "Additional context (optional)",
            placeholder=(
                "e.g. I saw their recent product launch, I have 5 years in fintech, "
                "referral from mutual connection..."
            ),
            height=120,
            key="extra_context",
        )

        if st.button("Generate Cold Email", type="primary", key="btn_generate"):
            if not config["target_company"] or config["target_company"] == "the company":
                st.error("Enter the target company name in the sidebar.")
            else:
                with st.spinner("CrewAI is writing your cold email..."):
                    try:
                        llm = get_llm(config["api_key"])
                        crew = build_cold_email_crew(
                            llm,
                            {**config, "extra_context": extra_context.strip()},
                        )
                        result = crew.kickoff()
                        st.session_state["last_cold_email"] = str(result)
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")

        if st.session_state.get("last_cold_email"):
            st.divider()
            st.subheader("Generated email")
            st.text_area(
                "Copy your email",
                value=st.session_state["last_cold_email"],
                height=320,
                key="cold_email_output",
            )

    with tab_reply:
        st.subheader("Reply to an incoming email")
        incoming_email = st.text_area(
            "Paste the email you received",
            placeholder="Paste the full email from the company here...",
            height=200,
            key="incoming_email",
        )
        reply_instructions = st.text_area(
            "Reply instructions (optional)",
            placeholder=(
                "e.g. Accept the meeting, ask for pricing, decline politely, "
                "request more details about the role..."
            ),
            height=80,
            key="reply_instructions",
        )

        if st.button("Generate Reply", type="primary", key="btn_reply"):
            if not incoming_email.strip():
                st.error("Paste the incoming email first.")
            else:
                with st.spinner("CrewAI is drafting your reply..."):
                    try:
                        llm = get_llm(config["api_key"])
                        crew = build_reply_crew(
                            llm,
                            {
                                **config,
                                "incoming_email": incoming_email.strip(),
                                "reply_instructions": reply_instructions.strip(),
                            },
                        )
                        result = crew.kickoff()
                        st.session_state["last_reply"] = str(result)
                    except Exception as exc:
                        st.error(f"Reply generation failed: {exc}")

        if st.session_state.get("last_reply"):
            st.divider()
            st.subheader("Generated reply")
            st.text_area(
                "Copy your reply",
                value=st.session_state["last_reply"],
                height=320,
                key="reply_output",
            )


if __name__ == "__main__":
    main()
