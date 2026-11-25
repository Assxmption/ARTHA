"""Real-data boundary tests for NSE option chains."""

def test_option_chain_unavailability_never_returns_synthetic_data(monkeypatch):
    import app.data.nse_fno as nse_fno

    monkeypatch.setattr(nse_fno, "fetch_option_chain_nselib", lambda symbol: None)
    monkeypatch.setattr(nse_fno, "fetch_option_chain_nse_direct", lambda symbol: None)

    result = nse_fno.get_option_chain("NIFTY")

    assert result["source"] == "none"
    assert result["available"] is False
    assert result["calls"] == []
    assert result["puts"] == []
    assert "mock" not in result.get("source", "")
