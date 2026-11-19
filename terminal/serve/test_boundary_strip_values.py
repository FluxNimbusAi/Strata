"""#5226/#5225: a string parameter's text arrives exactly as the model wrote it.

Both threads report a capture-boundary artifact - the model itself writes a doubled newline, a
wrapped value, a trailing space, or literal backslash-n text at the edge of a value - and the
threads' own reading of the shipped tree settled that serve passes the parser's argument string
through untouched: the capture owns the artifact end to end.  What this module pins is the serve
contract those probes named: the two template newlines that bracket a string value are dropped
and every other character is kept - whole, streamed, and chunked alike (the rule of #6032: the
parser never repairs a value, it reads what arrived).  With the mirror pinned, a model-versus-
serializer question on any of these shapes answers from the reply itself.  These are regression
cases for the current contract; the pack-side capture root the threads narrowed to stays open
and is not claimed here.

    python -m unittest serve.test_boundary_strip_values -v
"""
import json
import unittest

from serve.frontend import (CALL_END, CALL_START, FUNC_END, FUNC_START, PARAM_END, PARAM_START,
                            OutputParser, parse_tool_call)

BS = chr(92)      # one backslash, spelled so this file holds no escape sequences of its own
NL = chr(10)      # one line feed, likewise

SCHEMA = {"name": "write",
          "parameters": {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}}}

# the value shapes the two threads report, as the text the model writes for its string parameter
# between the template newline that opens the value and the template newline that closes it
VALUES = {
    "file keeps its own final newline": "print(1)" + NL,
    "captured newline rides on top": "print(1)" + NL + NL,
    "wrapped value": NL + "notes/weather-log.md" + NL,
    "trailing space variant": "notes/weather-log.md ",
    "backslash-n text kept verbatim": "printf('hi" + BS + "nthere')",
}


def call_body(pname, value):
    return (FUNC_START + "write>" + NL
            + PARAM_START + pname + ">" + NL + value + NL + PARAM_END + NL
            + FUNC_END)


def one_call(pname, value):
    return CALL_START + NL + call_body(pname, value) + NL + CALL_END


class BoundaryStripMirror(unittest.TestCase):
    """Only the two bracketing template newlines are lost; every character the model wrote
    between them survives, in both paths and at every chunk size."""

    def check_exact(self, text, expected):
        self.assertEqual(parse_tool_call(text[len(CALL_START):-len(CALL_END)], SCHEMA).arguments, expected)
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                    events = []
                    for i in range(0, len(text), size):
                        events.extend(parser.feed(text[i:i + size]))
                    events.extend(parser.finish())
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual([c.arguments for c in calls], [expected])
                    if stream:
                        self.assertEqual(json.loads("".join(e.text for e in events
                                                            if e.kind == "tool_args")), expected)

    def test_string_values_arrive_as_written(self):
        for label, value in VALUES.items():
            with self.subTest(label=label):
                self.check_exact(one_call("content", value), {"content": value})


class AdjacentTwoCalls(unittest.TestCase):
    """#5226's forced shape: both calls in one reply with nothing between them - the position
    the reports corrupt, which the shipped cases do not hold (its two-call case separates the
    calls with prose)."""

    def test_both_calls_exact(self):
        wrapped = VALUES["wrapped value"]
        text = one_call("path", "notes/weather-log.md") + NL + one_call("content", wrapped)
        expected = [{"path": "notes/weather-log.md"}, {"content": wrapped}]
        for stream in (False, True):
            for size in (1, 7, len(text)):
                with self.subTest(stream=stream, size=size):
                    parser = OutputParser(thinking=False, tools=[SCHEMA], stream_tools=stream)
                    events = []
                    for i in range(0, len(text), size):
                        events.extend(parser.feed(text[i:i + size]))
                    events.extend(parser.finish())
                    calls = [e.call for e in events if e.kind == "tool_call"]
                    self.assertEqual([c.arguments for c in calls], expected)
                    if stream:      # tool_start is a streaming event; the final parse carries only the calls
                        self.assertEqual(len([e for e in events if e.kind == "tool_start"]), 2)


if __name__ == "__main__":
    unittest.main()
