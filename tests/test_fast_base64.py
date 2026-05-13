from __future__ import annotations

import base64
import unittest

from mdexplore_app.fast_base64 import b64decode_loose, b64encode_ascii


class FastBase64Tests(unittest.TestCase):
    def test_b64encode_ascii_round_trips_and_has_canonical_length(self) -> None:
        sizes = [0, 1, 2, 3, 7, 31, 32, 33, 127, 256, 1023, 4096, 58943]
        for size in sizes:
            with self.subTest(size=size):
                payload = bytes((index % 251 for index in range(size)))
                encoded = b64encode_ascii(payload)
                expected_len = ((size + 2) // 3) * 4
                self.assertEqual(len(encoded), expected_len)
                self.assertEqual(base64.b64decode(encoded, validate=False), payload)

    def test_b64decode_loose_accepts_whitespace_and_missing_padding(self) -> None:
        raw = b"mdexplore-fast-base64"
        canonical = base64.b64encode(raw).decode("ascii")
        missing_padding = canonical.rstrip("=")
        with_whitespace = "\n".join(
            missing_padding[i : i + 5] for i in range(0, len(missing_padding), 5)
        )
        self.assertEqual(b64decode_loose(with_whitespace), raw)


if __name__ == "__main__":
    unittest.main()
