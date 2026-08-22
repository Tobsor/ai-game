import unittest
from argparse import Namespace
from unittest.mock import patch

import server


class ServerCliTests(unittest.TestCase):
    def test_parse_args_defaults(self):
        args = server.parse_args([])

        self.assertFalse(args.persist)
        self.assertFalse(args.verbose)

    def test_parse_args_supports_persist_and_verbose(self):
        args = server.parse_args(["--persist", "--verbose"])

        self.assertTrue(args.persist)
        self.assertTrue(args.verbose)

    @patch("server.uvicorn.run")
    @patch("server.configure_logging")
    @patch("server.parse_args")
    def test_main_forces_verbose_logging_when_flag_is_set(self, parse_args_mock, configure_logging_mock, uvicorn_run_mock):
        parse_args_mock.return_value = Namespace(persist=False, verbose=True)

        server.main()

        configure_logging_mock.assert_called_once_with("VERBOSE")
        self.assertFalse(server.app.state.persist_enabled)
        uvicorn_run_mock.assert_called_once()

    @patch.dict("os.environ", {"LOG_LEVEL": "DEBUG"}, clear=False)
    @patch("server.uvicorn.run")
    @patch("server.configure_logging")
    @patch("server.parse_args")
    def test_main_verbose_flag_overrides_environment_log_level(self, parse_args_mock, configure_logging_mock, uvicorn_run_mock):
        parse_args_mock.return_value = Namespace(persist=False, verbose=True)

        server.main()

        configure_logging_mock.assert_called_once_with("VERBOSE")
        uvicorn_run_mock.assert_called_once()

    @patch("server.uvicorn.run")
    @patch("server.configure_logging")
    @patch("server.parse_args")
    def test_main_enables_persistence_flag(self, parse_args_mock, configure_logging_mock, uvicorn_run_mock):
        parse_args_mock.return_value = Namespace(persist=True, verbose=False)

        server.main()

        configure_logging_mock.assert_called_once_with(None)
        self.assertTrue(server.app.state.persist_enabled)
        uvicorn_run_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
