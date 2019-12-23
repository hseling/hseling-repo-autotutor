import unittest

import hseling_api_autotutor


class HSELing_API_AutotutorTestCase(unittest.TestCase):

    def setUp(self):
        self.app = hseling_api_autotutor.app.test_client()

    def test_index(self):
        rv = self.app.get('/healthz')
        self.assertIn('Application Autotutor', rv.data.decode())


if __name__ == '__main__':
    unittest.main()
