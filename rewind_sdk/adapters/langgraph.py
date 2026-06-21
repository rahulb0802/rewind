def _message_classes():
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    except Exception:  # pragma: no cover - optional adapter dependency
        return None, None, None, None
    return AIMessage, HumanMessage, SystemMessage, ToolMessage


def _verification_halt_error_class():
    """Late import to avoid a circular dependency at module load time."""
    try:
        from ..verification import VerificationHaltError
        return VerificationHaltError
    except Exception:
        return None


def messages_to_dicts(messages):
    dicts = []
    for message in messages or []:
        if isinstance(message, dict):
            dicts.append(dict(message))
            continue

        AIMessage, HumanMessage, SystemMessage, ToolMessage = _message_classes()
        role = "user"
        if SystemMessage is not None and isinstance(message, SystemMessage):
            role = "system"
        elif AIMessage is not None and isinstance(message, AIMessage):
            role = "assistant"
        elif ToolMessage is not None and isinstance(message, ToolMessage):
            role = "tool"

        dicts.append(
            {
                "role": role,
                "content": getattr(message, "content", str(message)),
                "metadata": getattr(message, "tool_call_id", None),
                "name": getattr(message, "name", None),
                "id": getattr(message, "id", None),
                "tool_calls": getattr(message, "tool_calls", None),
                "additional_kwargs": getattr(message, "additional_kwargs", None),
                "response_metadata": getattr(message, "response_metadata", None),
            }
        )
    return dicts


def dicts_to_messages(dicts, message_format="langchain"):
    if message_format in ("dict", "dicts", None):
        return [dict(item) for item in dicts or []]

    AIMessage, HumanMessage, SystemMessage, ToolMessage = _message_classes()
    if HumanMessage is None:
        raise RuntimeError("langchain-core is required to convert dictionaries to messages.")

    messages = []
    for item in dicts or []:
        role = item.get("role", "user")
        content = item.get("content", "")
        kwargs = {
            "content": content,
            "name": item.get("name"),
            "id": item.get("id"),
            "additional_kwargs": item.get("additional_kwargs") or {},
            "response_metadata": item.get("response_metadata") or {},
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        if role == "system":
            messages.append(SystemMessage(**kwargs))
        elif role == "assistant":
            tool_calls = item.get("tool_calls")
            if tool_calls:
                kwargs["tool_calls"] = tool_calls
            messages.append(AIMessage(**kwargs))
        elif role == "tool":
            kwargs["tool_call_id"] = item.get("metadata") or item.get("tool_call_id") or "0"
            messages.append(ToolMessage(**kwargs))
        else:
            messages.append(HumanMessage(**kwargs))
    return messages


def infer_message_format(messages):
    for message in messages or []:
        if isinstance(message, dict):
            return "dict"
        return "langchain"
    return "dict"


class RewindLangGraph:
    def __init__(self, graph, session):
        self.graph = graph
        self.session = session

    def invoke(self, state, *args, **kwargs):
        self._update_memory(state)
        try:
            result = self.graph.invoke(state, *args, **kwargs)
        except Exception as exc:
            HaltError = _verification_halt_error_class()
            if HaltError is not None and isinstance(exc, HaltError):
                raise
            self.session.on_tool_result(error=exc)
            raise
        self._update_memory(result)
        return result

    def stream(self, state, *args, **kwargs):
        self._update_memory(state)
        try:
            for event in self.graph.stream(state, *args, **kwargs):
                # keeping rewind memory synced as framework makes new steps
                if isinstance(event, dict) and "messages" in event:
                    self._update_memory(event)
                for node_name, node_state in (event.items() if isinstance(event, dict) else []):
                    if isinstance(node_state, dict) and "messages" in node_state:
                        self._update_memory({"messages": node_state["messages"]})
                yield event
        except Exception as exc:
            HaltError = _verification_halt_error_class()
            if HaltError is not None and isinstance(exc, HaltError):
                raise
            self.session.on_tool_result(error=exc)
            raise

    def before_tool_node(self, state, tool_name=None):
        return self.session.on_tool_call(messages=(state or {}).get("messages", []), tool_name=tool_name)

    def after_tool_node(self, state, error=None):
        return self.session.on_tool_result(messages=(state or {}).get("messages", []), error=error)

    def _update_memory(self, state):
        if isinstance(state, dict) and "messages" in state:
            self.session.sync_memory(state["messages"])

    def __getattr__(self, name):
        return getattr(self.graph, name)


def wrap_langgraph(graph, session):
    return RewindLangGraph(graph, session=session)
