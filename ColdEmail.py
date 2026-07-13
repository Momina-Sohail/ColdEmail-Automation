import os
import re
import smtplib
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmailMessage:
    subject: str
    body: str
    recipient: str


@dataclass
class DeliveryStatus:
    success: bool
    message: str
    recipient: str
    subject: str
    email_body: str
    prepared_by_agent: str = ""


def _apply_groq_litellm_patch() -> None:
    """Remove cache_breakpoint from Groq calls (same fix as main.py)."""
    import litellm

    original_completion = litellm.completion

    def patched_completion(*args, **kwargs):
        if "messages" in kwargs:
            for msg in kwargs["messages"]:
                if isinstance(msg, dict) and "cache_breakpoint" in msg:
                    del msg["cache_breakpoint"]
        return original_completion(*args, **kwargs)

    litellm.completion = patched_completion


try:
    from crewai import Agent, Crew, LLM, Task

    _apply_groq_litellm_patch()
except ImportError as exc:
    raise ImportError(
        "Missing dependencies. Install from requirements.txt: "
        "streamlit, crewai, litellm, python-dotenv."
    ) from exc


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


def build_email_delivery_crew(
    llm: LLM, email_text: str, recipient: str, context: dict
) -> Crew:
    delivery_agent = Agent(
        role="Email Delivery Specialist",
        goal=(
            "Validate outbound emails and prepare them for reliable delivery "
            "with correct subject lines and professional formatting."
        ),
        backstory=(
            "You are an expert in business email delivery. You review messages "
            "for clarity, correct structure, and readiness to send. You fix "
            "only critical formatting issues without rewriting the author's voice."
        ),
        llm=llm,
        verbose=False,
    )

    task = Task(
        description=f"""
Review and prepare this email for delivery.

**Recipient:** {recipient}
**Sender:** {context["your_name"]} — {context["your_role"]} at {context["your_company"]}
**Target company:** {context["target_company"]}

**Email draft:**
---
{email_text}
---

Requirements:
- Ensure a clear subject line exists
- Body must not include the subject line
- Keep the original message intent and tone
- Fix only critical issues (missing subject, broken structure)

Output EXACTLY in this format with no extra sections:

SUBJECT: <subject line without "Subject:" prefix>
BODY:
<full email body starting with greeting>
""",
        expected_output=(
            "Structured email with SUBJECT and BODY sections ready for SMTP delivery."
        ),
        agent=delivery_agent,
    )

    return Crew(agents=[delivery_agent], tasks=[task], llm=llm, verbose=False)


def get_smtp_config() -> dict | None:
    host = os.getenv("SMTP_HOST", "").strip()
    port_raw = os.getenv("SMTP_PORT", "587").strip()
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "").strip()
    from_email = os.getenv("SMTP_FROM", username).strip()

    if not all([host, username, password]):
        return None

    try:
        port = int(port_raw)
    except ValueError:
        return None

    return {
        "host": host,
        "port": port,
        "username": username,
        "password": password,
        "from_email": from_email or username,
    }


def parse_structured_email(text: str) -> EmailMessage | None:
    match = re.search(
        r"SUBJECT:\s*(.+?)\s*\nBODY:\s*\n(.*)",
        text.strip(),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    subject = match.group(1).strip()
    body = match.group(2).strip()
    if subject and body:
        return EmailMessage(subject=subject, body=body, recipient="")
    return None


def parse_email_fallback(text: str) -> EmailMessage:
    lines = text.strip().splitlines()
    subject = "Follow-up"
    body_start = 0

    if lines:
        first = lines[0].strip()
        subject_match = re.match(r"^subject:\s*(.+)$", first, re.IGNORECASE)
        if subject_match:
            subject = subject_match.group(1).strip()
            body_start = 1

    body = "\n".join(lines[body_start:]).strip() or text.strip()
    return EmailMessage(subject=subject, body=body, recipient="")


def prepare_email_for_send(
    llm: LLM, email_text: str, recipient: str, context: dict
) -> tuple[EmailMessage, str]:
    crew = build_email_delivery_crew(llm, email_text, recipient, context)
    agent_output = str(crew.kickoff())
    parsed = parse_structured_email(agent_output)
    if parsed is None:
        parsed = parse_email_fallback(email_text)
    parsed.recipient = recipient
    return parsed, agent_output


def send_email_via_smtp(message: EmailMessage, smtp_config: dict) -> DeliveryStatus:
    msg = MIMEMultipart()
    msg["From"] = smtp_config["from_email"]
    msg["To"] = message.recipient
    msg["Subject"] = message.subject
    msg.attach(MIMEText(message.body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_config["host"], smtp_config["port"], timeout=30) as server:
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.sendmail(
                smtp_config["from_email"],
                [message.recipient],
                msg.as_string(),
            )
    except smtplib.SMTPException as exc:
        return DeliveryStatus(
            success=False,
            message=f"SMTP delivery failed: {exc}",
            recipient=message.recipient,
            subject=message.subject,
            email_body=message.body,
        )
    except OSError as exc:
        return DeliveryStatus(
            success=False,
            message=f"Could not connect to mail server: {exc}",
            recipient=message.recipient,
            subject=message.subject,
            email_body=message.body,
        )

    return DeliveryStatus(
        success=True,
        message=f"Email delivered successfully to {message.recipient}.",
        recipient=message.recipient,
        subject=message.subject,
        email_body=message.body,
    )


def generate_and_send(
    llm: LLM,
    crew_builder,
    crew_context: dict,
    recipient: str,
    smtp_config: dict,
) -> tuple[str, DeliveryStatus, str]:
    crew = crew_builder(llm, crew_context)
    generated = str(crew.kickoff())

    prepared, agent_output = prepare_email_for_send(
        llm, generated, recipient, crew_context
    )
    status = send_email_via_smtp(prepared, smtp_config)
    status.prepared_by_agent = agent_output
    return generated, status, agent_output
