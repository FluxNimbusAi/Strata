"""#5226: exact-JSON parity for the tool-call serializer, plus negative goldens pinning
that the two FALSIFIED candidate fixes (JSON-unescape; aggressive newline trim) would
corrupt legitimate values — so any future re-litigation trips these first.

    python -m unittest serve.test_serializer_parity_5226 -v

Parser-level and deterministic - no model, no GPU. Mirrors serve.test_quoted_call_values.
"""
import json
import unittest

from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START,
                            OutputParser, parse_tool_call)

SCHEMA = {"name": "write", "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}

# Values chosen so the falsified 'fixes' are decisive: every key must survive byte-exact.
VALUES = {
    "literal backslash-n stays literal": "line1\\nline2",
    "real newline stays": "line1\nline2",
    "escaped double quotes": 'He said \\"hi\\" loudly',
    "json-shaped payload verbatim": '{"a": 1, "b": "\\n"}',
    "trailing newline inside value": "keep\n",
    "unicode dash and tab": "ok\u00e9 \u2014 dash\ttab",
    "double backslash tail": "C:\\path\\",
    "empty value": "",
}


def call_text(value):
    return (f"{CALL_START}\n{FUNC_START}write>\n{PARAM_START}path>\nprobe.txt\n{PARAM_END}\n"
            f"{PARAM_START}content>\n{value}\n{PARAM_END}\n{FUNC_END}\n{CALL_END}")


class ExactJsonParity(unittest.TestCase):
    def test_parsed_arguments_are_byte_exact(self):
        for label, value in VALUES.items():
            with self.subTest(label=label):
                text = call_text(value)
                expected = {"path": "probe.txt", "content": value}
                got = parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments
                self.assertEqual(got, expected)
                self.assertEqual(json.dumps(got, sort_keys=True), json.dumps(expected, sort_keys=True))

    def test_stream_reconstruction_matches_parsed(self):
        for label, value in VALUES.items():
            text = call_text(value)
            expected = {"path": "probe.txt", "content": value}
            for stream in (False, True):
                for size in (1, 7, len(text)):
                    with self.subTest(label=label, stream=stream, size=size):
                        parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                        events = []
                        for i in range(0, len(text), size):
                            events.extend(parser.feed(text[i:i + size]))
                        events.extend(parser.finish())
                        calls = [e.call for e in events if e.kind == "tool_call"]
                        self.assertEqual([c.arguments for c in calls], [expected])
                        if stream:
                            self.assertEqual(json.loads("".join(e.text for e in events if e.kind == "tool_args")),
                                             expected)


class NegativeGoldens(unittest.TestCase):
    """Injection-negatives per BS-5.1a: prove the suite is NOT vacuous by running the
    falsified transforms and asserting they differ from what the shipped parser returns."""

    def test_json_unescape_mutation_is_caught(self):
        value = VALUES["literal backslash-n stays literal"]
        broken = value.encode("utf-8").decode("unicode_escape")  # the falsified 'fix'
        self.assertNotEqual(broken, value)  # mutation exists to be caught
        got = self._parse(value)
        self.assertEqual(got, value)        # shipped parser does NOT apply it
        self.assertNotEqual(got, broken)

    def test_aggressive_trim_mutation_is_caught(self):
        value = VALUES["trailing newline inside value"]
        broken = value.rstrip("\n")         # the falsified 'fix'
        self.assertNotEqual(broken, value)
        got = self._parse(value)
        self.assertEqual(got, value)
        self.assertNotEqual(got, broken)

    def test_backslash_tail_survives_unescape_attempt(self):
        value = VALUES["double backslash tail"]
        with self.assertRaises(Exception):   # trailing lone escape makes unicode_escape fail outright —
            value.encode("utf-8").decode("unicode_escape")  # proof the 'fix' cannot even be applied
        self.assertEqual(self._parse(value), value)

    @staticmethod
    def _parse(value):
        text = call_text(value)
        return parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments["content"]


if __name__ == "__main__":
    unittest.main()
