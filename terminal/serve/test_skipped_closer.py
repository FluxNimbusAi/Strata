"""#5756 (folded into the evolved param_end per the Flux signature ruling): a tool-call value
whose model skipped its own closer.  A close that matches a nested opener inside the value is
value text; when the value never got its own closer the value ends right before the first
declared boundary (FUNC_END, the wrapper CALL_END of the 4427 shape, or an opener of a
declared parameter not set yet, read through opener_at) - the old _skipped_closer_end helper
is gone.  The first ParserLevel cases are the three verbatim reproduction shapes from issue
5756; R4 pins the quoted-call rule and the element closer, R2 the declared-opener boundary.

    python -m unittest serve.test_skipped_closer     (from the terminal/ directory)
"""
import json
import unittest

from serve.frontend import (CALL_END, FUNC_END, PARAM_END, PARAM_START, OutputParser,
                            param_end, parse_tool_call)

PS, PE, FE, CE = PARAM_START, PARAM_END, FUNC_END, CALL_END
LT, SL = chr(60), chr(47)
FS_EDIT, FS_BASH = LT + "function=edit_file>", LT + "function=bash>"
EL, CEL = LT + "new_string>", LT + SL + "new_string>"
NL = chr(10)
S = {"type": "string"}


def tool(name, **props):
    return {"name": name, "parameters": {"type": "object", "properties": props}}


EDIT = tool("edit_file", path=S, old_string=S, new_string=S)
BASH = tool("bash", command=S)

# the three parser-level shapes from issue #5756 and the value each must keep
T1 = PS + "limit>25" + PE + NL + FE + NL + CE
W1 = PS + "limit>25" + PE + NL
T2 = PS + "limit>25" + PE + NL + PS + "old_string>x" + PE + NL + FE
W2 = PS + "limit>25" + PE + NL + PS + "old_string>x" + PE + NL
T3 = PS + "limit>25" + PE + NL + PE + NL + FE
W3 = PS + "limit>25" + PE + NL
REPRO = [(T1, W1), (T2, W2), (T3, W3)]


class ParserLevel(unittest.TestCase):
    def test_the_issue_three_cases(self):
        for text, want in REPRO:
            e = param_end(text, final=True, name="new_string")
            self.assertEqual(text[:e], want)

    def test_the_old_rule_stays_for_quoted_calls(self):
        # R4: skipped=False is the reasoning route rule for a quoted call: its FIRST
        # close ends the value, exactly main's old answer - the fold must not move it
        text, _ = REPRO[0]
        e = param_end(text, final=True, name="new_string", skipped=False)
        self.assertEqual(e, text.index(PE))

    def test_no_schema_means_no_declared_rule(self):
        # names=None: only a FUNC_END / CALL_END boundary ends the value, never an
        # unrelated opener (the declared-parameter rule is off without a schema)
        text, want = REPRO[1]
        e = param_end(text, final=True, name="new_string", names=None)
        self.assertEqual(text[:e], want)

    def test_declared_opener_ends_the_value(self):
        # R2 regression: an opener of a declared parameter not set yet ends the value,
        # through opener_at (the fold keeps no separate declared helper)
        e = param_end(T2, final=True, name="new_string", names={"path", "old_string"})
        self.assertEqual(T2[:e], W1)

    def test_element_closer_with_a_nested_pair_still_closes(self):
        # R4 regression: a nested pair inside an element value does not steal the
        # boundary; the element closer ends the value, consumed whole
        text = "a " + PS + "limit>25" + PE + " b" + EL + NL + FE
        e = param_end(text, final=True, name="new_string", closer=EL)
        self.assertEqual(text[:e], "a " + PS + "limit>25" + PE + " b")

    def test_streaming_holds_a_split_boundary_tag(self):
        # a closer or boundary split across windows must never end the value early
        for cut in range(1, 24):
            text = T1[:cut]
            e = param_end(text, final=False, name="new_string")
            self.assertLessEqual(e, 0, f"released early at cut {cut}: {text[e:]!r}")
        e = param_end(W1 + "~FU", final=False, name="new_string")
        self.assertEqual(e, -2)
        e = param_end(W1 + PS + "o", final=False, name="new_string", names={"old_string"})
        self.assertEqual(e, -2)


def parse_all(text, tools, **kw):
    p = OutputParser(thinking=False, tools=tools, recover=True, **kw)
    evs = p.feed(text) + p.finish("stop")
    return [(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"]


class EndToEnd(unittest.TestCase):
    def test_skipped_closer_and_next_declared_parameter(self):
        text = (CE + NL + FS_EDIT + NL + PS + "path>docs/tools.md" + PE + NL +
                PS + "new_string>" + NL +
                "line one" + NL + PS + "limit>25" + PE + NL + "line three" + NL +
                PS + "old_string>old" + PE + NL + FE + NL + CE)
        self.assertEqual(parse_all(text, [EDIT]),
                         [("edit_file", {"path": "docs/tools.md",
                                          "new_string": "line one" + NL + PS + "limit>25" + PE + NL + "line three",
                                          "old_string": "old"})])

    def test_two_calls_after_a_skipped_closer(self):
        # the batch route: a repaired call keeps its value whole AND the next call in
        # the same answer walks normally (multi-call wrappers are not a supported form)
        text = (FS_EDIT + NL + PS + "new_string>" + NL +
                PS + "limit>25" + PE + NL + FE + NL + FS_BASH + NL +
                PS + "command>ls" + PE + NL + FE)
        self.assertEqual(parse_all(text, [EDIT, BASH]),
                         [("edit_file", {"new_string": PS + "limit>25" + PE}),
                          ("bash", {"command": "ls"})])

    def test_a_complete_nested_pair_still_closes_normally(self):
        text = (CE + NL + FS_EDIT + NL + PS + "new_string>" + NL +
                PS + "limit>25" + PE + NL + "text" + PE + NL + FE + NL + CE)
        self.assertEqual(parse_all(text, [EDIT]),
                         [("edit_file", {"new_string": PS + "limit>25" + PE + NL + "text"})])

    def test_batch_route_without_a_schema(self):
        # the parse route with schema=None: the declared rule is off, the boundary
        # before the function close still keeps the whole value
        body = FS_EDIT + NL + PS + "new_string>" + NL + T2
        call = parse_tool_call(body, None)
        self.assertEqual(call.name, "edit_file")
        self.assertEqual(call.arguments["new_string"], W2.strip(NL))


class Streaming(unittest.TestCase):
    # batch form (two bare calls): multi-call `<tool_call>` wrappers were never
    # a supported shape, neither before nor after the fold
    TWO_CALLS = (FS_EDIT + NL + PS + "new_string>" + NL +
                 PS + "limit>25" + PE + NL + FE + NL + FS_BASH + NL +
                 PS + "command>ls" + PE + NL + FE)
    EXPECT = [{"new_string": PS + "limit>25" + PE}, {"command": "ls"}]

    def test_deltas_decode_to_the_final_call_at_every_split(self):
        # the issue chunk sweep: every split at every size, the streamed argument
        # deltas decode to exactly the final call (nothing shown twice)
        for size in range(1, 25):
            p = OutputParser(thinking=False, tools=[EDIT, BASH], stream_tools=True, recover=True)
            evs = []
            for i in range(0, len(self.TWO_CALLS), size):
                evs += p.feed(self.TWO_CALLS[i:i + size])
            evs += p.finish("stop")
            buckets, cur, finals = [], None, []
            for e in evs:
                if e.kind == "tool_start":
                    cur = []
                    buckets.append(cur)
                elif e.kind == "tool_args":
                    cur.append(e.text)
                elif e.kind == "tool_call":
                    finals.append((e.call.name, e.call.arguments))
            self.assertEqual(finals, [("edit_file", self.EXPECT[0]), ("bash", self.EXPECT[1])], f"size {size}")
            for b, want in zip(buckets, self.EXPECT):
                got = "".join(b)
                self.assertEqual(got, json.dumps(want, separators=(",", ":")),
                                 f"size {size}: streamed {got!r} != final")


if __name__ == "__main__":
    unittest.main()
