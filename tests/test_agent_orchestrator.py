"""End-to-end orchestrator tests with faked retriever + generator (no network/DB).

The fake generator dispatches on a distinctive substring of each node's system
prompt (router / decomposer / synthesizer / fact-checker) so one object can
stand in across the whole pipeline, the same way a real multi-purpose model
would be routed to different prompts by the orchestrator itself.
"""

from __future__ import annotations

from sec_10k_agent.agent.orchestrator import run_agent
from sec_10k_agent.agent.synthesize import NO_INFO_TEXT
from sec_10k_agent.rag.models import LLMResponse
from sec_10k_agent.retrieval.models import RetrievedChunk

ROUTE_SINGLE = """{
  "needs_decomposition": false, "retrieval_modes": ["semantic"], "is_temporal": false,
  "candidate_filters": {"tickers": ["AAPL"], "fiscal_years": [2024], "sections": ["Item 1A"]},
  "reasoning": "single company narrative question"
}"""

SYNTH_ONE_CLAIM = """{"claims": [
  {"text": "Apple discloses supply chain concentration risk.", "source_ids": ["c1"]}
]}"""


class _FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(self, query: str, **kwargs: object) -> list[RetrievedChunk]:
        return self._chunks


class _ScriptedGenerator:
    """Dispatches by system-prompt substring; verify replies come from a queue
    so retry tests can script "fails then succeeds"."""

    def __init__(
        self,
        route_reply: str = ROUTE_SINGLE,
        decompose_reply: str = "",
        synth_reply: str = SYNTH_ONE_CLAIM,
        verify_replies: list[str] | None = None,
    ) -> None:
        self._route_reply = route_reply
        self._decompose_reply = decompose_reply
        self._synth_reply = synth_reply
        self._verify_replies = list(verify_replies or ['{"supported": true}'])
        self.calls: list[str] = []

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> LLMResponse:
        if "query router" in system:
            self.calls.append("route")
            text = self._route_reply
        elif "query decomposer" in system:
            self.calls.append("decompose")
            text = self._decompose_reply
        elif "financial analyst assistant" in system:
            self.calls.append("synthesize")
            text = self._synth_reply
        elif "strict fact-checker" in system:
            self.calls.append("verify")
            text = (
                self._verify_replies.pop(0)
                if len(self._verify_replies) > 1
                else self._verify_replies[0]
            )
        else:
            raise AssertionError(f"unrecognized system prompt: {system[:50]!r}")
        return LLMResponse(text=text, model="fake")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1",
        ticker="AAPL",
        fiscal_year=2024,
        section="Item 1A",
        text="Supply chain text.",
        distance=0.1,
    )


def test_single_hop_happy_path_no_decomposition_no_retry() -> None:
    retriever = _FakeRetriever([_chunk()])
    generator = _ScriptedGenerator()
    state = run_agent("What supply chain risks does Apple disclose?", retriever, generator)  # type: ignore[arg-type]

    assert state.routing_plan is not None
    assert state.routing_plan.needs_decomposition is False
    assert len(state.subqueries) == 1
    assert state.retry_count == 0
    assert state.verified_claims[0].verified is True
    assert "Apple discloses supply chain concentration risk." in state.final_answer
    assert "[Source: AAPL FY2024 10-K, Item 1A]" in state.final_answer
    # decompose was never called (no LLM hit) since needs_decomposition is False.
    assert "decompose" not in generator.calls


def test_retry_on_failed_verification_then_succeeds() -> None:
    retriever = _FakeRetriever([_chunk()])
    generator = _ScriptedGenerator(verify_replies=['{"supported": false}', '{"supported": true}'])
    state = run_agent("What supply chain risks does Apple disclose?", retriever, generator)  # type: ignore[arg-type]

    assert state.retry_count == 1
    assert state.verified_claims[0].verified is True
    assert state.final_answer != NO_INFO_TEXT
    assert generator.calls.count("verify") == 2
    assert generator.calls.count("synthesize") == 2  # re-synthesized on retry


def test_gives_up_after_max_retries_returns_abstention() -> None:
    retriever = _FakeRetriever([_chunk()])
    generator = _ScriptedGenerator(verify_replies=['{"supported": false}'])
    state = run_agent("q", retriever, generator, max_retries=1)  # type: ignore[arg-type]

    assert state.retry_count == 1  # stopped at max_retries
    assert state.verified_claims[0].verified is False
    assert state.final_answer == NO_INFO_TEXT


def test_no_retrieval_short_circuits_to_abstention() -> None:
    retriever = _FakeRetriever([])  # nothing retrieved
    generator = _ScriptedGenerator(synth_reply='{"claims": []}')
    state = run_agent("an unanswerable question", retriever, generator)  # type: ignore[arg-type]
    assert state.final_answer == NO_INFO_TEXT


def test_structured_xbrl_subquery_logs_tool_call() -> None:
    route_reply = """{
      "needs_decomposition": false, "retrieval_modes": ["structured_xbrl"], "is_temporal": false,
      "candidate_filters": {"tickers": ["AAPL"], "fiscal_years": [2024], "sections": []},
      "concept_hint": "total revenue",
      "reasoning": "asks for a number"
    }"""
    retriever = _FakeRetriever([])
    generator = _ScriptedGenerator(route_reply=route_reply, synth_reply='{"claims": []}')
    state = run_agent(
        "What was Apple's revenue in FY2024?",
        retriever,  # type: ignore[arg-type]
        generator,  # type: ignore[arg-type]
        xbrl_lookup=lambda **kw: [],
    )
    assert len(state.tool_calls) == 1
    assert "lookup_financial_metric" in state.tool_calls[0]
    assert "total revenue" in state.tool_calls[0]
