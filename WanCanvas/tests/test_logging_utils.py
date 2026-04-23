from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from wancanvas.utils.logging import read_json_report, write_json_report


class LoggingUtilsTest(unittest.TestCase):
    def test_read_json_report_reads_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_json_report(Path(tmpdir) / 'demo.json', {'status': True, 'value': 3})

            payload = read_json_report(path)

            self.assertEqual(payload, {'status': True, 'value': 3})

    def test_read_json_report_retries_after_transient_decode_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'transient.json'
            path.write_text('{"status": ', encoding='utf-8')

            def repair() -> None:
                time.sleep(0.05)
                write_json_report(path, {'status': 'ready'})

            thread = threading.Thread(target=repair)
            thread.start()
            try:
                payload = read_json_report(path, retries=10, delay_sec=0.02)
            finally:
                thread.join()

            self.assertEqual(payload, {'status': 'ready'})

    def test_read_json_report_retries_after_transient_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'late.json'

            def create() -> None:
                time.sleep(0.05)
                write_json_report(path, {'ok': True})

            thread = threading.Thread(target=create)
            thread.start()
            try:
                payload = read_json_report(path, retries=10, delay_sec=0.02)
            finally:
                thread.join()

            self.assertEqual(payload, {'ok': True})


if __name__ == '__main__':
    unittest.main()
