"""
Offline operational tests.
Simulates a disabled network environment to verify that datasets,
tokenizers, models, and diacritic engines function with zero external connectivity.
"""

import socket
import pytest
import config
from src.diacritics.restore_diacritics import SindhiDiacriticRestorer
from src.data.validate_dataset import DatasetValidator
import pandas as pd


@pytest.fixture
def disconnect_network(monkeypatch):
    """Simulates an offline environment by blocking outbound socket connections."""
    def guarded_connect(*args, **kwargs):
        raise OSError("Simulated Network Disconnection: System is strictly offline!")
    monkeypatch.setattr(socket, "create_connection", guarded_connect)


class TestOfflineOperation:
    """Verifies that all local components operate without network access."""

    def test_diacritic_engine_offline(self, disconnect_network):
        """Tests that diacritic model restores text while completely disconnected."""
        restorer = SindhiDiacriticRestorer()
        res = restorer.restore("علم ۽ عقل")
        assert res["marked"] is not None
        assert len(res["marked"]) > 0

    def test_local_dataset_loading_offline(self, disconnect_network):
        """Verifies local dataset splits are readable offline."""
        train_file = config.PARALLEL_TRAIN_FILE
        assert train_file.exists(), f"Local train file missing: {train_file}"
        df = pd.read_csv(train_file)
        assert len(df) > 0

    def test_data_validator_offline(self, disconnect_network):
        """Verifies data cleaner executes offline."""
        validator = DatasetValidator()
        norm = validator.normalize_text("  مان سنڌي آهيان۔  ")
        assert norm == "مان سنڌي آهيان۔"
        assert validator.contains_arabic_script(norm)

    def test_model_directory_configuration(self):
        """Verifies configured model directory points to local filesystem."""
        assert str(config.MODELS_DIR).startswith(str(config.BASE_DIR))
        assert config.DIACRITIC_MODEL_FILE.exists()
