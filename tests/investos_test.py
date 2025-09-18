import investos as inv


def test_version():
    assert inv.__version__ == "0.7.0"


def test_key_and_endpoint():
    assert isinstance(inv.api_key, str)
    assert isinstance(inv.api_endpoint, str) and inv.api_endpoint.startswith("http")
