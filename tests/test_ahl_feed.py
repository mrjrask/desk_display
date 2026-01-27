from data_fetch import _sanitize_ahl_payload


def test_sanitize_ahl_payload_strips_guard_prefixes():
    raw = "while(1);\n\ufeff{\"foo\": \"bar\"}"
    assert _sanitize_ahl_payload(raw) == '{"foo": "bar"}'


def test_sanitize_ahl_payload_handles_comments_and_whitespace():
    raw = "/* comment */  \n  {\n  \"ok\": true\n}"
    assert _sanitize_ahl_payload(raw) == '{\n  "ok": true\n}'
