def test_import():
    import src.legal_pipeline as lp
    assert hasattr(lp, "__all__")
