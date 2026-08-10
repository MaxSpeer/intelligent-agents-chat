"""NiceGUI chat interface backed by SQLite and selectable model profiles."""

from __future__ import annotations

import asyncio
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from importlib.metadata import version
import logging
import platform
from time import monotonic
from uuid import uuid4

from nicegui import app as nicegui_app, ui

from intelligent_agents_chat.config import Settings
from intelligent_agents_chat.database import (
    DEFAULT_CONVERSATION_TITLE,
    DEFAULT_PROJECT_ID,
    ChatRepository,
    Conversation,
    Message,
    Project,
)
from intelligent_agents_chat.llm import LLMError, VLLMGateway
from intelligent_agents_chat.logging_config import (
    configure_logging,
    log_event,
    sanitized_endpoint,
    shutdown_logging,
)


settings = Settings.from_env()
configure_logging(settings)
logger = logging.getLogger(__name__)
log_event(
    logger,
    logging.INFO,
    "application.runtime.loaded",
    python_version=platform.python_version(),
    operating_system=platform.platform(),
    dependency_versions={
        "nicegui": version("nicegui"),
        "openai": version("openai"),
    },
)
log_event(
    logger,
    logging.INFO,
    "application.configuration.loaded",
    database_path=str(settings.database_path),
    default_profile_key=settings.default_profile_key,
    profile_count=len(settings.model_profiles),
    profiles=[
        {
            "key": profile.key,
            "backend": profile.backend,
            "model": profile.model,
            "endpoint": sanitized_endpoint(profile.base_url),
            "supports_thinking": profile.supports_thinking,
        }
        for profile in settings.model_profiles
    ],
    api_authentication_configured=settings.api_key not in {"", "not-needed"},
    request_timeout_seconds=settings.request_timeout_seconds,
    max_tokens=settings.max_tokens,
    thinking_max_tokens=settings.thinking_max_tokens,
    temperature=settings.temperature,
    system_prompt_chars=len(settings.system_prompt),
)
repository = ChatRepository(settings.database_path)
try:
    repository.initialize()
except Exception:
    logger.exception(
        "application.database_initialization_failed",
        extra={
            "event": "application.database_initialization_failed",
            "database_path": str(settings.database_path),
        },
    )
    raise
gateway = VLLMGateway(settings)
active_generations: set[str] = set()
log_event(logger, logging.INFO, "application.initialized")


def _log_unhandled_application_exception(error: Exception) -> None:
    logger.error(
        "application.unhandled_exception",
        exc_info=(type(error), error, error.__traceback__),
        extra={
            "event": "application.unhandled_exception",
            "error_type": type(error).__name__,
        },
    )


nicegui_app.on_exception(_log_unhandled_application_exception)


ui.add_head_html(
    """
    <style>
        :root {
            --ink: #172033;
            --muted: #657089;
            --primary: #635bff;
            --primary-dark: #4b43df;
            --surface: rgba(255, 255, 255, 0.88);
            --surface-solid: #ffffff;
            --line: rgba(104, 117, 147, 0.16);
            --sidebar: rgba(245, 246, 251, 0.92);
        }

        html, body, #q-app {
            min-height: 100%;
        }

        .nicegui-content {
            padding: 0 !important;
        }

        body {
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 5%, rgba(99, 91, 255, .13), transparent 28rem),
                radial-gradient(circle at 88% 90%, rgba(38, 198, 218, .11), transparent 32rem),
                #f8f9fc;
        }

        .app-shell {
            display: grid;
            grid-template-columns: 290px minmax(0, 1fr);
            width: 100%;
            height: 100vh;
            overflow: hidden;
        }

        .sidebar {
            min-width: 0;
            height: 100vh;
            padding: 1.25rem 1rem;
            gap: 1rem;
            overflow: hidden;
            border-right: 1px solid var(--line);
            background: var(--sidebar);
            backdrop-filter: blur(22px);
        }

        .brand-mark {
            display: grid;
            width: 2.35rem;
            height: 2.35rem;
            place-items: center;
            border-radius: .85rem;
            color: white;
            background: linear-gradient(135deg, var(--primary), #857fff);
            box-shadow: 0 10px 24px rgba(99, 91, 255, .28);
        }

        .brand-name {
            font-size: .96rem;
            font-weight: 800;
            letter-spacing: -.02em;
        }

        .project-picker {
            width: 100%;
            flex-wrap: nowrap;
            gap: .45rem;
            align-items: center;
        }

        .project-select {
            min-width: 0;
            flex: 1 1 auto;
        }

        .project-create {
            flex: 0 0 auto;
            color: var(--primary) !important;
            background: rgba(99, 91, 255, .09) !important;
        }

        .eyebrow {
            color: var(--muted);
            font-size: .68rem;
            font-weight: 700;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .new-chat-button {
            width: 100%;
            min-height: 2.8rem;
            border-radius: .9rem;
            color: white !important;
            background: var(--primary) !important;
            box-shadow: 0 10px 24px rgba(99, 91, 255, .22);
        }

        .conversation-list {
            min-height: 0;
            flex: 1 1 auto;
            gap: .4rem;
            overflow-y: auto;
            padding-right: .2rem;
        }

        .conversation-item {
            width: 100%;
            min-width: 0;
            flex-wrap: nowrap;
            gap: .25rem;
            align-items: center;
        }

        .conversation-select {
            min-width: 0;
            flex: 1 1 auto;
            justify-content: flex-start;
            overflow: hidden;
            border-radius: .78rem;
            color: var(--muted) !important;
        }

        .conversation-select .q-btn__content {
            display: block;
            overflow: hidden;
            text-align: left;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .conversation-select.active {
            color: var(--primary-dark) !important;
            background: rgba(99, 91, 255, .1) !important;
        }

        .conversation-delete {
            color: #9aa2b5 !important;
            opacity: 0;
            transition: opacity .15s ease, color .15s ease;
        }

        .conversation-item:hover .conversation-delete,
        .conversation-delete:focus {
            opacity: 1;
        }

        .conversation-delete:hover {
            color: #d14f68 !important;
        }

        .sidebar-footer {
            padding: .9rem;
            border: 1px solid var(--line);
            border-radius: .9rem;
            background: rgba(255, 255, 255, .58);
        }

        .main-panel {
            min-width: 0;
            height: 100vh;
            gap: 0;
            overflow: hidden;
        }

        .chat-header {
            z-index: 2;
            width: 100%;
            min-height: 5rem;
            flex-wrap: nowrap;
            padding: .85rem clamp(1rem, 3vw, 2rem);
            border-bottom: 1px solid var(--line);
            background: rgba(255, 255, 255, .72);
            backdrop-filter: blur(18px);
        }

        .chat-title {
            max-width: min(42vw, 38rem);
            overflow: hidden;
            font-size: 1.05rem;
            font-weight: 750;
            letter-spacing: -.02em;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .model-select {
            width: min(15rem, 27vw);
        }

        .thinking-toggle {
            flex: 0 0 auto;
            padding: .15rem .5rem;
            border: 1px solid var(--line);
            border-radius: .8rem;
            color: var(--muted);
            background: rgba(255, 255, 255, .58);
        }

        .status-dot {
            width: .5rem;
            height: .5rem;
            border-radius: 999px;
            background: #20b486;
            box-shadow: 0 0 0 5px rgba(32, 180, 134, .11);
        }

        .message-scroll {
            width: 100%;
            min-height: 0;
            flex: 1 1 auto;
        }

        .message-column {
            width: min(100%, 900px);
            min-height: 100%;
            margin: 0 auto;
            padding: 2rem clamp(1rem, 4vw, 2.5rem) 1.5rem;
            gap: 1.15rem;
        }

        .chat-message {
            width: 100%;
        }

        .chat-message .q-message-text {
            max-width: min(44rem, 82vw);
            border-radius: 1rem;
            box-shadow: 0 8px 22px rgba(34, 42, 74, .07);
        }

        .chat-message .q-message-text-content {
            line-height: 1.58;
        }

        .q-message-sent .q-message-text {
            color: white;
            background: linear-gradient(135deg, var(--primary), #7770ff) !important;
        }

        .q-message-received .q-message-text {
            color: var(--ink);
            border: 1px solid var(--line);
            background: var(--surface-solid) !important;
        }

        .streaming-content {
            max-width: min(42rem, 76vw);
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.58;
        }

        .empty-state {
            width: min(100%, 36rem);
            margin: auto;
            padding: 2.5rem;
            align-items: center;
            gap: .8rem;
            text-align: center;
            border: 1px solid rgba(255, 255, 255, .9);
            border-radius: 1.5rem;
            background: var(--surface);
            box-shadow: 0 20px 52px rgba(34, 42, 74, .09);
            backdrop-filter: blur(18px);
        }

        .empty-icon {
            display: grid;
            width: 3.4rem;
            height: 3.4rem;
            place-items: center;
            border-radius: 1.1rem;
            color: var(--primary);
            background: rgba(99, 91, 255, .1);
        }

        .composer-area {
            z-index: 2;
            width: 100%;
            padding: .75rem clamp(1rem, 4vw, 2.5rem) 1.2rem;
            background: linear-gradient(to top, #f8f9fc 72%, rgba(248, 249, 252, 0));
        }

        .composer-row {
            width: min(100%, 850px);
            min-height: 3.7rem;
            margin: 0 auto;
            flex-wrap: nowrap;
            gap: .4rem;
            padding: .45rem .5rem .45rem 1rem;
            border: 1px solid var(--line);
            border-radius: 1.15rem;
            background: white;
            box-shadow: 0 14px 36px rgba(34, 42, 74, .11);
        }

        .composer-input {
            min-width: 0;
            flex: 1 1 auto;
        }

        .composer-input .q-field__control:before,
        .composer-input .q-field__control:after {
            border: 0 !important;
        }

        .send-button {
            color: white !important;
            background: var(--primary) !important;
        }

        .stop-button {
            color: #d14f68 !important;
            background: rgba(209, 79, 104, .1) !important;
        }

        @media (max-width: 720px) {
            .app-shell {
                grid-template-columns: 1fr;
                grid-template-rows: 15rem minmax(0, 1fr);
            }

            .sidebar {
                width: 100%;
                height: 15rem;
                padding: .9rem 1rem;
                gap: .7rem;
                border-right: 0;
                border-bottom: 1px solid var(--line);
            }

            .conversation-list {
                flex-direction: row;
                overflow-x: auto;
                overflow-y: hidden;
            }

            .conversation-delete {
                opacity: 1;
            }

            .conversation-item {
                width: 13rem;
                min-width: 13rem;
            }

            .sidebar-footer {
                display: none;
            }

            .main-panel {
                height: calc(100vh - 15rem);
            }

            .chat-header {
                min-height: 4.4rem;
            }

            .status-copy {
                display: none;
            }
        }
    </style>
    """,
    shared=True,
)


@dataclass(slots=True)
class PageState:
    """State that must remain local to one connected browser page."""

    project_id: str
    conversation_id: str
    page_id: str = field(default_factory=lambda: str(uuid4()))
    generating: bool = False
    stop_event: asyncio.Event | None = None
    generation_task: asyncio.Task[None] | None = None
    generation_id: str | None = None


def _conversation_title(message: str, limit: int = 44) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3].rstrip()}..."


def _display_time(value: datetime) -> str:
    return value.astimezone().strftime("%H:%M")


def _profile_label(profile_key: str | None) -> str:
    if profile_key is None:
        return "Assistant"
    try:
        return settings.profile(profile_key).label
    except KeyError:
        return profile_key


def _completion_messages(messages: list[Message]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if settings.system_prompt:
        result.append({"role": "system", "content": settings.system_prompt})
    result.extend({"role": message.role, "content": message.content} for message in messages)
    return result


@ui.page("/")
def index() -> None:
    """Render a client-local chat workspace backed by shared SQLite persistence."""
    projects = repository.list_projects()
    initial_project = next(
        (project for project in projects if project.id == DEFAULT_PROJECT_ID),
        projects[0],
    )
    conversations = repository.list_conversations(initial_project.id)
    initial = (
        conversations[0]
        if conversations
        else repository.create_conversation(
            settings.default_profile_key,
            project_id=initial_project.id,
        )
    )
    if initial.model_profile not in settings.profile_options:
        repository.set_model_profile(initial.id, settings.default_profile_key)
        normalized_initial = repository.get_conversation(initial.id)
        if normalized_initial is None:
            raise RuntimeError("Conversation disappeared while updating its model profile")
        initial = normalized_initial
    state = PageState(project_id=initial_project.id, conversation_id=initial.id)

    def page_event(level: int, event: str, **fields: object) -> None:
        context: dict[str, object] = {
            "page_id": state.page_id,
            "project_id": state.project_id,
            "conversation_id": state.conversation_id,
        }
        if state.generation_id is not None:
            context["generation_id"] = state.generation_id
        context.update(fields)
        log_event(logger, level, event, **context)

    page_event(
        logging.INFO,
        "ui.page.opened",
        available_project_count=len(projects),
        initial_model_profile=initial.model_profile,
    )

    def current_project() -> Project:
        project = repository.get_project(state.project_id)
        if project is not None:
            return project

        available_projects = repository.list_projects()
        if not available_projects:
            page_event(logging.ERROR, "ui.project.recovery_failed")
            raise RuntimeError("No project is available")
        project = available_projects[0]
        page_event(
            logging.WARNING,
            "ui.project.recovered",
            missing_project_id=state.project_id,
            replacement_project_id=project.id,
        )
        state.project_id = project.id
        return project

    def current_conversation() -> Conversation:
        conversation = repository.get_conversation(state.conversation_id)
        if conversation is None or conversation.project_id != state.project_id:
            missing_conversation_id = state.conversation_id
            conversations = repository.list_conversations(state.project_id)
            conversation = (
                conversations[0]
                if conversations
                else repository.create_conversation(
                    settings.default_profile_key,
                    project_id=state.project_id,
                )
            )
            state.conversation_id = conversation.id
            page_event(
                logging.WARNING,
                "ui.conversation.recovered",
                missing_conversation_id=missing_conversation_id,
                replacement_conversation_id=conversation.id,
            )
        if conversation.model_profile not in settings.profile_options:
            previous_model_profile = conversation.model_profile
            repository.set_model_profile(conversation.id, settings.default_profile_key)
            conversation = repository.get_conversation(conversation.id)
            if conversation is None:
                raise RuntimeError("Conversation disappeared while updating its model profile")
            page_event(
                logging.WARNING,
                "ui.conversation.model_normalized",
                previous_model_profile=previous_model_profile,
                replacement_model_profile=settings.default_profile_key,
            )
        profile = settings.profile(conversation.model_profile)
        if conversation.thinking_enabled and not profile.supports_thinking:
            repository.set_thinking_enabled(conversation.id, False)
            normalized_conversation = repository.get_conversation(conversation.id)
            if normalized_conversation is None:
                raise RuntimeError("Conversation disappeared while disabling thinking")
            conversation = normalized_conversation
            page_event(
                logging.WARNING,
                "ui.conversation.thinking_normalized",
                model_profile=profile.key,
                thinking_enabled=False,
            )
        return conversation

    def set_busy(is_busy: bool) -> None:
        state.generating = is_busy
        page_event(logging.DEBUG, "ui.busy_state.changed", is_busy=is_busy)
        if is_busy:
            composer.disable()
            send_button.disable()
            stop_button.enable()
            model_select.disable()
            thinking_toggle.disable()
            status_label.set_text("Generating")
        else:
            composer.enable()
            send_button.enable()
            stop_button.disable()
            model_select.enable()
            profile = settings.profile(current_conversation().model_profile)
            if profile.supports_thinking:
                thinking_toggle.enable()
            else:
                thinking_toggle.disable()
            status_label.set_text("Ready")
            composer.run_method("focus")

    def render_project_picker() -> None:
        project = current_project()
        project_select.set_options(
            {item.id: item.name for item in repository.list_projects()},
            value=project.id,
        )

    def render_conversation_list() -> None:
        conversation_list.clear()
        with conversation_list:
            for conversation in repository.list_conversations(state.project_id):
                active_class = " active" if conversation.id == state.conversation_id else ""
                with ui.row().classes("conversation-item"):
                    ui.button(
                        conversation.title,
                        icon="chat_bubble_outline",
                        on_click=partial(select_conversation, conversation.id),
                    ).props("flat no-caps align=left").classes(
                        f"conversation-select{active_class}"
                    ).tooltip(
                        f"{conversation.title} - updated {_display_time(conversation.updated_at)}"
                    )
                    delete_button = (
                        ui.button(
                            icon="delete_outline",
                            on_click=partial(confirm_delete_conversation, conversation.id),
                        )
                        .props("flat round dense")
                        .classes("conversation-delete")
                    )
                    delete_button.props["aria-label"] = f"Delete {conversation.title}"
                    delete_button.tooltip("Delete conversation")

    def render_header() -> None:
        conversation = current_conversation()
        profile = settings.profile(conversation.model_profile)
        title_label.set_text(conversation.title)
        model_select.set_options(
            settings.profile_options,
            value=conversation.model_profile,
        )
        thinking_toggle.set_value(conversation.thinking_enabled)
        if profile.supports_thinking and not state.generating:
            thinking_toggle.enable()
        else:
            thinking_toggle.disable()

    def render_messages() -> None:
        messages_container.clear()
        messages = repository.list_messages(state.conversation_id)
        with messages_container:
            if not messages:
                with ui.column().classes("empty-state"):
                    with ui.element("div").classes("empty-icon"):
                        ui.icon("forum", size="md")
                    ui.label("Start a conversation").classes("text-xl font-bold")
                    ui.label(
                        f'Messages in "{current_project().name}" are kept separate and saved '
                        "locally in SQLite."
                    ).classes("text-sm text-slate-500 leading-relaxed")
                    ui.label(
                        "Use the offline test model or an OpenAI-compatible vLLM profile"
                    ).classes("text-xs text-slate-400")
            else:
                for message in messages:
                    sent = message.role == "user"
                    name = "You" if sent else _profile_label(message.model_profile)
                    ui.chat_message(
                        text=message.content,
                        name=name,
                        stamp=_display_time(message.created_at),
                        sent=sent,
                    ).classes("chat-message")
        message_scroll.scroll_to(percent=1)

    def render_all() -> None:
        render_project_picker()
        render_conversation_list()
        render_header()
        render_messages()

    def change_project(event) -> None:
        project_id = str(event.value)
        if project_id == state.project_id:
            page_event(logging.DEBUG, "ui.project.switch_ignored", reason="already_selected")
            return
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.project.switch_blocked",
                target_project_id=project_id,
                reason="generation_active",
            )
            ui.notify("Stop the current response before switching projects.", type="warning")
            render_project_picker()
            return
        if repository.get_project(project_id) is None:
            page_event(
                logging.WARNING,
                "ui.project.switch_blocked",
                target_project_id=project_id,
                reason="project_missing",
            )
            render_project_picker()
            return

        previous_project_id = state.project_id
        previous_conversation_id = state.conversation_id
        state.project_id = project_id
        conversations = repository.list_conversations(project_id)
        conversation = (
            conversations[0]
            if conversations
            else repository.create_conversation(
                settings.default_profile_key,
                project_id=project_id,
            )
        )
        state.conversation_id = conversation.id
        page_event(
            logging.INFO,
            "ui.project.switched",
            previous_project_id=previous_project_id,
            previous_conversation_id=previous_conversation_id,
            model_profile=conversation.model_profile,
        )
        render_all()

    async def create_project() -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.project.create_blocked",
                reason="generation_active",
            )
            ui.notify("Stop the current response before creating a project.", type="warning")
            return

        page_event(logging.DEBUG, "ui.project.create_dialog_opened")

        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
            ui.label("Create project").classes("text-xl font-bold")
            ui.label("Conversations in different projects stay separate.").classes(
                "text-sm text-slate-500"
            )
            project_name_input = (
                ui.input(
                    label="Project name",
                    placeholder="e.g. Research",
                )
                .props("outlined autofocus")
                .classes("w-full")
            )
            project_name_input.on(
                "keydown.enter",
                lambda: dialog.submit(str(project_name_input.value or "")),
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button(
                    "Create",
                    on_click=lambda: dialog.submit(str(project_name_input.value or "")),
                ).props("unelevated no-caps")

        project_name = await dialog
        if project_name is None or project_name is False:
            page_event(logging.DEBUG, "ui.project.create_cancelled")
            return
        try:
            project = repository.create_project(str(project_name))
        except ValueError as error:
            page_event(
                logging.WARNING,
                "ui.project.create_rejected",
                reason=str(error),
                project_name_chars=len(str(project_name)),
            )
            ui.notify(str(error), type="warning")
            return

        conversation = repository.create_conversation(
            settings.default_profile_key,
            project_id=project.id,
        )
        state.project_id = project.id
        state.conversation_id = conversation.id
        page_event(
            logging.INFO,
            "ui.project.created_and_selected",
            project_name_chars=len(project.name),
            model_profile=conversation.model_profile,
        )
        render_all()
        composer.run_method("focus")

    def select_conversation(conversation_id: str) -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.conversation.switch_blocked",
                target_conversation_id=conversation_id,
                reason="generation_active",
            )
            ui.notify("Stop the current response before switching chats.", type="warning")
            return
        conversation = repository.get_conversation(conversation_id)
        if conversation is None or conversation.project_id != state.project_id:
            page_event(
                logging.WARNING,
                "ui.conversation.switch_blocked",
                target_conversation_id=conversation_id,
                reason=("conversation_missing" if conversation is None else "different_project"),
            )
            render_all()
            return
        previous_conversation_id = state.conversation_id
        state.conversation_id = conversation_id
        page_event(
            logging.INFO,
            "ui.conversation.switched",
            previous_conversation_id=previous_conversation_id,
            model_profile=conversation.model_profile,
        )
        render_all()

    def new_conversation() -> None:
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.conversation.create_blocked",
                reason="generation_active",
            )
            ui.notify("Stop the current response before starting a new chat.", type="warning")
            return
        conversation = current_conversation()
        if conversation.title == DEFAULT_CONVERSATION_TITLE and not repository.list_messages(
            conversation.id
        ):
            page_event(
                logging.DEBUG,
                "ui.conversation.empty_reused",
                model_profile=conversation.model_profile,
            )
            composer.run_method("focus")
            return
        created = repository.create_conversation(
            settings.default_profile_key,
            project_id=state.project_id,
        )
        previous_conversation_id = state.conversation_id
        state.conversation_id = created.id
        page_event(
            logging.INFO,
            "ui.conversation.created_and_selected",
            previous_conversation_id=previous_conversation_id,
            model_profile=created.model_profile,
        )
        render_all()

    async def confirm_delete_conversation(conversation_id: str) -> None:
        if conversation_id in active_generations:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="generation_active",
            )
            ui.notify("Stop the response in this chat before deleting it.", type="warning")
            return
        conversation = repository.get_conversation(conversation_id)
        if conversation is None:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="conversation_missing",
            )
            render_all()
            return

        page_event(
            logging.DEBUG,
            "ui.conversation.delete_dialog_opened",
            target_conversation_id=conversation_id,
            title_chars=len(conversation.title),
        )

        with ui.dialog() as dialog, ui.card().classes("w-96 max-w-full p-6 gap-5"):
            ui.label("Delete conversation?").classes("text-xl font-bold")
            ui.label(f'"{conversation.title}" and all of its messages will be removed.').classes(
                "text-sm text-slate-500"
            )
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button("Delete", on_click=lambda: dialog.submit(True)).props(
                    "unelevated no-caps color=negative"
                )

        if not await dialog:
            page_event(
                logging.DEBUG,
                "ui.conversation.delete_cancelled",
                target_conversation_id=conversation_id,
            )
            return
        if conversation_id in active_generations:
            page_event(
                logging.WARNING,
                "ui.conversation.delete_blocked",
                target_conversation_id=conversation_id,
                reason="generation_started_while_confirming",
            )
            ui.notify("This chat started generating and cannot be deleted yet.", type="warning")
            return
        deleted = repository.delete_conversation(conversation_id)
        was_selected = conversation_id == state.conversation_id
        replacement_conversation_id = None
        if was_selected:
            remaining = repository.list_conversations(state.project_id)
            replacement = (
                remaining[0]
                if remaining
                else repository.create_conversation(
                    settings.default_profile_key,
                    project_id=state.project_id,
                )
            )
            state.conversation_id = replacement.id
            replacement_conversation_id = replacement.id
        page_event(
            logging.INFO,
            "ui.conversation.deleted",
            target_conversation_id=conversation_id,
            deleted=deleted,
            was_selected=was_selected,
            replacement_conversation_id=replacement_conversation_id,
        )
        render_all()

    def change_model(event) -> None:
        if state.conversation_id in active_generations:
            page_event(
                logging.WARNING,
                "ui.model.change_blocked",
                target_model_profile=str(event.value),
                reason="generation_active",
            )
            ui.notify("Stop this chat's response before changing its model.", type="warning")
            render_header()
            return
        profile_key = str(event.value)
        try:
            profile = settings.profile(profile_key)
        except KeyError:
            page_event(
                logging.WARNING,
                "ui.model.change_blocked",
                target_model_profile=profile_key,
                reason="unknown_profile",
            )
            render_header()
            return
        conversation = current_conversation()
        updated = repository.set_model_profile(state.conversation_id, profile_key)
        thinking_disabled = False
        if conversation.thinking_enabled and not profile.supports_thinking:
            repository.set_thinking_enabled(state.conversation_id, False)
            thinking_disabled = True
        page_event(
            logging.INFO,
            "ui.model.changed",
            previous_model_profile=conversation.model_profile,
            model_profile=profile_key,
            updated=updated,
            thinking_disabled=thinking_disabled,
        )
        render_conversation_list()
        render_header()

    def change_thinking(event) -> None:
        requested = bool(event.value)
        conversation = current_conversation()
        profile = settings.profile(conversation.model_profile)
        if conversation.id in active_generations:
            page_event(
                logging.WARNING,
                "ui.thinking.change_blocked",
                requested_thinking_enabled=requested,
                reason="generation_active",
            )
            ui.notify("Stop this chat's response before changing Thinking.", type="warning")
            render_header()
            return
        if requested == conversation.thinking_enabled:
            page_event(
                logging.DEBUG,
                "ui.thinking.change_ignored",
                thinking_enabled=requested,
                reason="already_selected",
            )
            return
        if requested and not profile.supports_thinking:
            page_event(
                logging.WARNING,
                "ui.thinking.change_blocked",
                requested_thinking_enabled=requested,
                model_profile=profile.key,
                reason="profile_unsupported",
            )
            ui.notify("The selected model profile does not support Thinking.", type="warning")
            render_header()
            return

        updated = repository.set_thinking_enabled(conversation.id, requested)
        page_event(
            logging.INFO,
            "ui.thinking.changed",
            model_profile=profile.key,
            thinking_enabled=requested,
            max_tokens=(settings.thinking_max_tokens if requested else settings.max_tokens),
            updated=updated,
        )
        render_conversation_list()

    def stop_generation() -> None:
        if state.stop_event is None:
            page_event(logging.DEBUG, "ui.generation.stop_ignored", reason="not_generating")
            return

        task_was_active = state.generation_task is not None and not state.generation_task.done()
        page_event(
            logging.INFO,
            "ui.generation.stop_requested",
            generation_task_active=task_was_active,
        )
        state.stop_event.set()
        stop_button.disable()
        status_label.set_text("Stopping")
        if task_was_active:
            assert state.generation_task is not None
            state.generation_task.cancel()

    async def send_message() -> None:
        raw_text = str(composer.value or "")
        text = raw_text.strip()
        if not text:
            page_event(
                logging.DEBUG,
                "ui.generation.rejected",
                reason="empty_message",
                input_message_chars=len(raw_text),
            )
            ui.notify("Write a message first.", type="warning")
            return
        if state.generating:
            page_event(
                logging.WARNING,
                "ui.generation.rejected",
                reason="page_generation_active",
                input_message_chars=len(text),
            )
            return

        conversation = current_conversation()
        if conversation.id in active_generations:
            page_event(
                logging.WARNING,
                "ui.generation.rejected",
                reason="conversation_generation_active",
                input_message_chars=len(text),
            )
            ui.notify("This chat is already generating in another tab.", type="warning")
            return

        generation_id = str(uuid4())
        generation_started_at = monotonic()
        state.generation_id = generation_id
        active_generations.add(conversation.id)

        try:
            previous_messages = repository.list_messages(conversation.id)
            profile = settings.profile(conversation.model_profile)
            page_event(
                logging.INFO,
                "ui.generation.started",
                model_profile=profile.key,
                model_backend=profile.backend,
                model_name=profile.model,
                thinking_enabled=conversation.thinking_enabled,
                max_tokens=(
                    settings.thinking_max_tokens
                    if conversation.thinking_enabled
                    else settings.max_tokens
                ),
                input_message_chars=len(text),
                previous_message_count=len(previous_messages),
            )
            repository.add_message(conversation.id, "user", text)
            if conversation.title == DEFAULT_CONVERSATION_TITLE and not previous_messages:
                repository.rename_conversation(conversation.id, _conversation_title(text))

            composer.value = ""
            render_all()
            set_busy(True)
            if conversation.thinking_enabled:
                status_label.set_text("Thinking")
            stop_event = asyncio.Event()
            state.stop_event = stop_event
            state.generation_task = asyncio.current_task()

            with messages_container:
                with ui.chat_message(name=profile.label, sent=False).classes("chat-message"):
                    progress_text = (
                        f"Thinking with {profile.label}..."
                        if conversation.thinking_enabled
                        else f"Generating with {profile.label}..."
                    )
                    assistant_label = ui.label(progress_text).classes("streaming-content")
            message_scroll.scroll_to(percent=1)
        except Exception:
            logger.exception(
                "ui.generation.preparation_failed",
                extra={
                    "event": "ui.generation.preparation_failed",
                    "page_id": state.page_id,
                    "project_id": state.project_id,
                    "conversation_id": conversation.id,
                    "generation_id": generation_id,
                    "model_profile": conversation.model_profile,
                    "input_message_chars": len(text),
                },
            )
            try:
                state.stop_event = None
                state.generation_task = None
                active_generations.discard(conversation.id)
                if state.generating:
                    set_busy(False)
            finally:
                state.generation_id = None
            raise

        chunks: list[str] = []
        stopped = False
        outcome = "streaming"
        last_paint = monotonic()
        try:
            persisted_messages = repository.list_messages(conversation.id)
            completion_messages = _completion_messages(persisted_messages)
            page_event(
                logging.DEBUG,
                "ui.generation.request_prepared",
                model_profile=profile.key,
                thinking_enabled=conversation.thinking_enabled,
                completion_message_count=len(completion_messages),
                completion_chars=sum(
                    len(message.get("content", "")) for message in completion_messages
                ),
            )
            async with aclosing(
                gateway.stream_reply(
                    profile,
                    completion_messages,
                    request_id=generation_id,
                    thinking_enabled=conversation.thinking_enabled,
                )
            ) as stream:
                async for chunk in stream:
                    if stop_event.is_set():
                        stopped = True
                        break
                    chunks.append(chunk)
                    now = monotonic()
                    if now - last_paint >= 0.04:
                        assistant_label.set_text("".join(chunks))
                        message_scroll.scroll_to(percent=1)
                        last_paint = now

            if not "".join(chunks).strip() and not stopped:
                if conversation.thinking_enabled:
                    raise LLMError(
                        f"{profile.label} produced no final answer. Its thinking token budget "
                        "may be exhausted; disable Thinking or increase "
                        "VLLM_THINKING_MAX_TOKENS."
                    )
                raise LLMError(f"{profile.label} returned an empty response.")
            outcome = "stopped" if stopped else "completed"
        except asyncio.CancelledError:
            if not stop_event.is_set():
                outcome = "unexpected_cancellation"
                page_event(
                    logging.ERROR,
                    "ui.generation.cancelled_unexpectedly",
                    model_profile=profile.key,
                )
                raise
            stopped = True
            outcome = "stopped"
        except LLMError as error:
            outcome = "model_error"
            page_event(
                logging.WARNING,
                "ui.generation.model_error",
                model_profile=profile.key,
                thinking_enabled=conversation.thinking_enabled,
                error_type=type(error).__name__,
            )
            ui.notify(str(error), type="negative", multi_line=True, timeout=8000)
        except Exception as error:
            outcome = "unexpected_error"
            logger.exception(
                "ui.generation.unexpected_error",
                extra={
                    "event": "ui.generation.unexpected_error",
                    "page_id": state.page_id,
                    "project_id": state.project_id,
                    "conversation_id": conversation.id,
                    "generation_id": generation_id,
                    "model_profile": profile.key,
                    "thinking_enabled": conversation.thinking_enabled,
                    "error_type": type(error).__name__,
                },
            )
            raise
        finally:
            content = "".join(chunks).strip()
            assistant_message_saved = False
            try:
                if content:
                    repository.add_message(
                        conversation.id,
                        "assistant",
                        content,
                        model_profile=profile.key,
                    )
                    assistant_message_saved = True
                if stopped:
                    ui.notify(
                        "Generation stopped; the partial response was saved."
                        if content
                        else "Stopped.",
                        type="info",
                    )
            except Exception as error:
                outcome = "persistence_error"
                logger.exception(
                    "ui.generation.persistence_failed",
                    extra={
                        "event": "ui.generation.persistence_failed",
                        "page_id": state.page_id,
                        "project_id": state.project_id,
                        "conversation_id": conversation.id,
                        "generation_id": generation_id,
                        "model_profile": profile.key,
                        "thinking_enabled": conversation.thinking_enabled,
                        "error_type": type(error).__name__,
                        "output_chars": len(content),
                    },
                )
                raise
            finally:
                page_event(
                    logging.INFO,
                    "ui.generation.finished",
                    model_profile=profile.key,
                    thinking_enabled=conversation.thinking_enabled,
                    outcome=outcome,
                    stopped=stopped,
                    stream_chunk_count=len(chunks),
                    output_chars=len(content),
                    assistant_message_saved=assistant_message_saved,
                    duration_ms=round((monotonic() - generation_started_at) * 1000, 2),
                )
                state.stop_event = None
                state.generation_task = None
                active_generations.discard(conversation.id)
                try:
                    set_busy(False)
                    if repository.get_conversation(state.conversation_id) is not None:
                        render_all()
                finally:
                    state.generation_id = None

    with ui.element("main").classes("app-shell"):
        with ui.column().classes("sidebar"):
            with ui.row().classes("w-full items-center gap-3 px-1"):
                with ui.element("div").classes("brand-mark"):
                    ui.icon("auto_awesome", size="sm")
                with ui.column().classes("gap-0"):
                    ui.label("Agent Lab").classes("brand-name")
                    ui.label("Project workspace").classes("eyebrow")

            with ui.row().classes("project-picker"):
                project_select = (
                    ui.select(
                        {project.id: project.name for project in projects},
                        value=state.project_id,
                        label="Project",
                        on_change=change_project,
                    )
                    .props("outlined dense options-dense")
                    .classes("project-select")
                )
                project_create_button = (
                    ui.button(
                        icon="create_new_folder",
                        on_click=create_project,
                    )
                    .props("flat round dense")
                    .classes("project-create")
                )
                project_create_button.props["aria-label"] = "Create project"
                project_create_button.tooltip("Create project")

            ui.button(
                "New conversation",
                icon="add",
                on_click=new_conversation,
            ).props("unelevated no-caps").classes("new-chat-button")

            ui.label("Project chats").classes("eyebrow px-2 pt-1")
            conversation_list = ui.column().classes("conversation-list w-full")

            with ui.column().classes("sidebar-footer gap-1"):
                ui.label("Local history").classes("text-sm font-semibold")
                ui.label("SQLite persistence enabled").classes("text-xs text-slate-500")

        with ui.column().classes("main-panel"):
            with ui.row().classes("chat-header items-center justify-between"):
                with ui.column().classes("min-w-0 gap-0"):
                    ui.label("Conversation").classes("eyebrow")
                    title_label = ui.label().classes("chat-title")
                with ui.row().classes("items-center gap-4 no-wrap"):
                    model_select = (
                        ui.select(
                            settings.profile_options,
                            value=initial.model_profile,
                            label="Model",
                            on_change=change_model,
                        )
                        .props("outlined dense options-dense")
                        .classes("model-select")
                    )
                    thinking_toggle = (
                        ui.switch(
                            "Thinking",
                            value=initial.thinking_enabled,
                            on_change=change_thinking,
                        )
                        .props("dense color=deep-purple")
                        .classes("thinking-toggle")
                    )
                    thinking_toggle.tooltip(
                        f"Off: up to {settings.max_tokens:,} output tokens; "
                        f"on: up to {settings.thinking_max_tokens:,}"
                    )
                    with ui.row().classes("items-center gap-3 no-wrap"):
                        ui.element("span").classes("status-dot")
                        status_label = ui.label("Ready").classes(
                            "status-copy text-sm text-slate-500"
                        )
                        status_label.props["role"] = "status"
                        status_label.props["aria-live"] = "polite"

            message_scroll = ui.scroll_area().classes("message-scroll")
            with message_scroll:
                messages_container = ui.column().classes("message-column")

            with ui.column().classes("composer-area gap-2"):
                with ui.row().classes("composer-row items-center"):
                    composer = (
                        ui.input(
                            placeholder="Message the model...",
                        )
                        .props("borderless autocomplete=off")
                        .classes("composer-input")
                    )
                    composer.props["aria-label"] = "Message"
                    composer.on("keydown.enter", send_message)
                    stop_button = (
                        ui.button(
                            icon="stop",
                            on_click=stop_generation,
                        )
                        .props("round unelevated")
                        .classes("stop-button")
                    )
                    stop_button.props["aria-label"] = "Stop generation"
                    stop_button.disable()
                    stop_button.tooltip("Stop generation")
                    send_button = (
                        ui.button(
                            icon="arrow_upward",
                            on_click=send_message,
                        )
                        .props("round unelevated")
                        .classes("send-button")
                    )
                    send_button.props["aria-label"] = "Send message"
                    send_button.tooltip("Send message")
                ui.label(
                    "Enter to send - responses and conversation history are stored locally"
                ).classes("w-full text-center text-xs text-slate-400")

    render_all()
    ui.context.client.on_disconnect(
        lambda: page_event(
            logging.INFO,
            "ui.page.disconnected",
            generation_active=state.generating,
        )
    )


def main() -> None:
    """Start the NiceGUI development server."""
    log_event(
        logger,
        logging.INFO,
        "application.server.starting",
        host="127.0.0.1",
        port=8080,
        reload=False,
    )
    try:
        ui.run(title="Agent Lab", favicon="✨", host="127.0.0.1", port=8080, reload=False)
    except Exception:
        logger.exception(
            "application.server.failed",
            extra={"event": "application.server.failed"},
        )
        raise
    finally:
        log_event(logger, logging.INFO, "application.server.stopped")
        shutdown_logging()


if __name__ in {"__main__", "__mp_main__"}:
    main()
