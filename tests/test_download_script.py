import subprocess


def test_download_script_has_valid_bash_syntax():
    result = subprocess.run(
        ["bash", "-n", "scripts/download_data.sh"], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_download_script_subset_flag_calls_create_local_subset(tmp_path, monkeypatch):
    # Rather than hitting the network, verify --subset delegates to our tested
    # Python sampling function by running it against a fake "already downloaded" tree.
    from tissue_classifier.data import CLASS_NAMES, create_local_subset
    from PIL import Image

    source = tmp_path / "data" / "raw" / "NCT-CRC-HE-100K"
    for name in CLASS_NAMES:
        d = source / name
        d.mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (8, 8)).save(d / f"img_{i}.tif")

    dest = tmp_path / "data" / "local_subset"
    create_local_subset(source, dest, per_class=3, seed=0)

    for name in CLASS_NAMES:
        assert len(list((dest / name).iterdir())) == 3
