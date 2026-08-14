from __future__ import annotations

import unittest

import run_agent_task


class AgentTaskEncodingTests(unittest.TestCase):
    def test_codex_streams_are_utf8_and_decode_errors_cannot_kill_pump(self) -> None:
        options = run_agent_task._agent_text_stream_options()
        self.assertIs(options["text"], True)
        self.assertEqual(options["encoding"], "utf-8")
        self.assertEqual(options["errors"], "replace")
        self.assertEqual(options["bufsize"], 1)


if __name__ == "__main__":
    unittest.main()
