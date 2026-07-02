def test_package_importable():
    import tissue_classifier
    assert tissue_classifier.__version__ == "0.1.0"
