#!/usr/bin/env python3

import unittest
from pathlib import Path


class CurlLargeImagePayloadTest(unittest.TestCase):
    def test_image_payloads_use_local_http_urls_instead_of_base64(self):
        script = Path(__file__).parents[1] / "curl.sh"
        text = script.read_text(encoding="utf-8")

        self.assertNotIn("base64", text.lower())
        self.assertNotIn("data:image", text)
        self.assertIn("python3 -m http.server", text)
        self.assertIn("trap cleanup EXIT INT TERM", text)
        self.assertEqual(text.count('--arg uri "$IMAGE_URL"'), 1)
        self.assertEqual(text.count("send_image_request \"Test "), 2)
        self.assertIn("outdoor-courtyard.png", text)
        self.assertIn("indoor-kitchen.png", text)


if __name__ == "__main__":
    unittest.main()
