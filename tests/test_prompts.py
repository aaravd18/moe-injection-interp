import pytest

from moe_interp.dataset import CharSpan
from moe_interp.prompts import PromptRenderError, char_span_to_token_span, render_prompt


def test_render_matches_chat_template_and_keeps_boundaries(toy_tokenizer):
    msgs = [{"role": "system", "content": "Sys."}, {"role": "user", "content": "Hello\nworld"}]
    rp = render_prompt(toy_tokenizer, msgs)
    assert rp.text == "<|endoftext|><|system|>\nSys.\n<|user|>\nHello\nworld\n<|assistant|>\n"
    assert rp.input_ids[0] == toy_tokenizer.bos_token_id
    assert len(rp.attention_mask) == len(rp.input_ids)
    for m, (s, e) in zip(msgs, rp.message_char_spans):
        assert rp.text[s:e] == m["content"]
    ts = rp.message_token_spans[1]
    assert toy_tokenizer.decode(rp.input_ids[ts.start:ts.end]) == "Hello\nworld" and ts.exact


def test_without_generation_prompt(toy_tokenizer):
    rp = render_prompt(toy_tokenizer, [{"role": "user", "content": "x"}], add_generation_prompt=False)
    assert not rp.text.endswith("<|assistant|>\n")


def test_dropped_role_is_detected(toy_tokenizer):
    with pytest.raises(PromptRenderError, match="role='tool'"):
        render_prompt(toy_tokenizer, [{"role": "user", "content": "q"}, {"role": "tool", "content": "RESULT 42"}])


def test_span_mapping(toy_tokenizer):
    msgs = [{"role": "user", "content": "Doc here. Question: why?"}]
    rp = render_prompt(toy_tokenizer, msgs, char_spans={"q": CharSpan(0, 10, 24)})
    sp = rp.spans["q"]
    assert toy_tokenizer.decode(rp.input_ids[sp.start:sp.end]) == "Question: why?"
    assert sp.exact


def test_char_span_to_token_span_inexact_boundary():
    offsets = [(0, 0), (0, 3), (3, 9), (9, 12)]  # special token, "abc", " defgh", "ijk"
    ts = char_span_to_token_span(offsets, 4, 9)
    assert (ts.start, ts.end, ts.exact) == (2, 3, False)
    assert char_span_to_token_span(offsets, 20, 25) is None


def test_missing_chat_template_raises(toy_tokenizer):
    import copy
    tok = copy.deepcopy(toy_tokenizer)
    tok.chat_template = None
    with pytest.raises(PromptRenderError, match="chat_template"):
        render_prompt(tok, [{"role": "user", "content": "x"}])
