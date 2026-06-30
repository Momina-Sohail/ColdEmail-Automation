import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

try:
    from ColdEmail import (
        DeliveryStatus,
        build_cold_email_crew,
        build_reply_crew,
        generate_and_send,
        get_llm,
        get_smtp_config,
    )
except ImportError as exc:
    st.error(
        "Could not load email agents. Install dependencies from `requirements.txt` "
        "and run this app from the `COLD EMAIL` folder."
    )
    st.code(str(exc))
    st.stop()


def init_session_state() -> None:
    defaults = {
        "last_cold_email": "",
        "last_cold_status": None,
        "last_reply": "",
        "last_reply_status": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_delivery_status(status: DeliveryStatus | None) -> None:
    if status is None:
        return

    st.markdown("### Delivery status")
    if status.success:
        st.success(status.message)
    else:
        st.error(status.message)

    col_to, col_subject = st.columns(2)
    with col_to:
        st.markdown(f"**To:** {status.recipient}")
    with col_subject:
        st.markdown(f"**Subject:** {status.subject}")


def render_sidebar() -> dict | None:
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Used by CrewAI to generate emails.",
        )

        st.divider()
        st.subheader("Email delivery")
        smtp_config = get_smtp_config()
        if smtp_config:
            st.success("SMTP ready (from `.env`)")
        else:
            st.warning("Add SMTP_HOST, SMTP_USERNAME, and SMTP_PASSWORD to `.env`.")

        auto_send = st.checkbox(
            "Send automatically after generation",
            value=True,
        )

    if not api_key.strip():
        st.info("Add your Groq API key in the sidebar to get started.")
        return None

    return {
        "api_key": api_key.strip(),
        "smtp_config": smtp_config,
        "auto_send": auto_send,
    }


def render_cold_email_tab(settings: dict) -> None:
    st.markdown("Enter your details, generate a personalized email, and send it in one step.")

    col_you, col_target = st.columns(2)

    with col_you:
        st.subheader("Your details")
        your_company = st.text_input("Company", placeholder="Acme Solutions")
        your_name = st.text_input("Your name", placeholder="Jane Doe")
        your_role = st.text_input("Your role", placeholder="Founder")
        your_offer = st.text_area(
            "What you offer",
            placeholder="Brief value proposition",
            height=100,
        )

    with col_target:
        st.subheader("Recipient")
        target_company = st.text_input("Target company", placeholder="Stripe")
        target_contact = st.text_input("Contact name (optional)", placeholder="Alex Chen")
        target_role = st.text_input("Contact role (optional)", placeholder="Head of Partnerships")
        recipient_email = st.text_input(
            "Recipient email",
            value=os.getenv("DEFAULT_RECIPIENT_EMAIL", ""),
            placeholder="alex@company.com",
        )

    extra_context = st.text_area(
        "Extra context (optional)",
        placeholder="Recent news, mutual connection, specific angle...",
        height=100,
    )

    if st.button("Generate & send email", type="primary", use_container_width=True):
        if not target_company.strip():
            st.error("Target company is required.")
            return
        if settings["auto_send"] and not recipient_email.strip():
            st.error("Recipient email is required for automatic sending.")
            return
        if settings["auto_send"] and not settings["smtp_config"]:
            st.error("SMTP is not configured. Update your `.env` file or disable auto-send.")
            return

        context = {
            "your_company": your_company.strip() or "Your Company",
            "your_name": your_name.strip() or "Your Name",
            "your_role": your_role.strip() or "Your Role",
            "your_offer": your_offer.strip() or "Our product or service",
            "target_company": target_company.strip(),
            "target_contact": target_contact.strip(),
            "target_role": target_role.strip(),
            "extra_context": extra_context.strip(),
        }

        with st.status("Running email agents...", expanded=True) as status_ui:
            try:
                llm = get_llm(settings["api_key"])
                status_ui.write("Writing personalized cold email...")
                if settings["auto_send"]:
                    status_ui.write("Preparing delivery and sending via SMTP...")
                    generated, delivery, _ = generate_and_send(
                        llm,
                        build_cold_email_crew,
                        context,
                        recipient_email.strip(),
                        settings["smtp_config"],
                    )
                    st.session_state["last_cold_email"] = generated
                    st.session_state["last_cold_status"] = delivery
                    status_ui.update(
                        label="Done" if delivery.success else "Send failed",
                        state="complete" if delivery.success else "error",
                    )
                else:
                    crew = build_cold_email_crew(llm, context)
                    st.session_state["last_cold_email"] = str(crew.kickoff())
                    st.session_state["last_cold_status"] = None
                    status_ui.update(label="Email generated", state="complete")
            except Exception as exc:
                status_ui.update(label="Failed", state="error")
                st.error(f"Something went wrong: {exc}")

    if st.session_state["last_cold_email"]:
        st.divider()
        st.markdown("### Generated email")
        st.text_area(
            "Email preview",
            value=st.session_state["last_cold_email"],
            height=280,
            label_visibility="collapsed",
        )
        render_delivery_status(st.session_state["last_cold_status"])


def render_reply_tab(settings: dict) -> None:
    st.markdown("Paste an incoming email and generate a reply that sends automatically.")

    col_you, col_target = st.columns(2)
    with col_you:
        your_company = st.text_input("Company", key="reply_company", placeholder="Acme Solutions")
        your_name = st.text_input("Your name", key="reply_name", placeholder="Jane Doe")
        your_role = st.text_input("Your role", key="reply_role", placeholder="Founder")
        your_offer = st.text_area(
            "What you offer",
            key="reply_offer",
            placeholder="Brief value proposition",
            height=80,
        )
    with col_target:
        target_company = st.text_input(
            "Company you are replying to",
            key="reply_target_company",
            placeholder="Stripe",
        )
        recipient_email = st.text_input(
            "Reply-to email",
            key="reply_recipient",
            value=os.getenv("DEFAULT_RECIPIENT_EMAIL", ""),
            placeholder="alex@company.com",
        )

    incoming_email = st.text_area(
        "Incoming email",
        placeholder="Paste the full email you received...",
        height=180,
    )
    reply_instructions = st.text_area(
        "Reply instructions (optional)",
        placeholder="e.g. Accept the meeting, ask for pricing...",
        height=80,
    )

    if st.button("Generate & send reply", type="primary", use_container_width=True):
        if not incoming_email.strip():
            st.error("Paste the incoming email first.")
            return
        if settings["auto_send"] and not recipient_email.strip():
            st.error("Reply-to email is required for automatic sending.")
            return
        if settings["auto_send"] and not settings["smtp_config"]:
            st.error("SMTP is not configured. Update your `.env` file or disable auto-send.")
            return

        context = {
            "your_company": your_company.strip() or "Your Company",
            "your_name": your_name.strip() or "Your Name",
            "your_role": your_role.strip() or "Your Role",
            "your_offer": your_offer.strip() or "Our product or service",
            "target_company": target_company.strip() or "the company",
            "incoming_email": incoming_email.strip(),
            "reply_instructions": reply_instructions.strip(),
        }

        with st.status("Running reply agents...", expanded=True) as status_ui:
            try:
                llm = get_llm(settings["api_key"])
                status_ui.write("Drafting reply...")
                if settings["auto_send"]:
                    status_ui.write("Preparing delivery and sending via SMTP...")
                    generated, delivery, _ = generate_and_send(
                        llm,
                        build_reply_crew,
                        context,
                        recipient_email.strip(),
                        settings["smtp_config"],
                    )
                    st.session_state["last_reply"] = generated
                    st.session_state["last_reply_status"] = delivery
                    status_ui.update(
                        label="Done" if delivery.success else "Send failed",
                        state="complete" if delivery.success else "error",
                    )
                else:
                    crew = build_reply_crew(llm, context)
                    st.session_state["last_reply"] = str(crew.kickoff())
                    st.session_state["last_reply_status"] = None
                    status_ui.update(label="Reply generated", state="complete")
            except Exception as exc:
                status_ui.update(label="Failed", state="error")
                st.error(f"Something went wrong: {exc}")

    if st.session_state["last_reply"]:
        st.divider()
        st.markdown("### Generated reply")
        st.text_area(
            "Reply preview",
            value=st.session_state["last_reply"],
            height=280,
            label_visibility="collapsed",
        )
        render_delivery_status(st.session_state["last_reply_status"])


def main() -> None:
    st.set_page_config(
        page_title="Cold Email Agent",
        page_icon="✉️",
        layout="wide",
    )

    init_session_state()
    st.title("Cold Email Agent")
    st.caption("Generate outreach emails with AI and send them automatically.")

    settings = render_sidebar()
    if settings is None:
        return

    tab_cold, tab_reply = st.tabs(["Cold outreach", "Reply to email"])
    with tab_cold:
        render_cold_email_tab(settings)
    with tab_reply:
        render_reply_tab(settings)


if __name__ == "__main__":
    main()
