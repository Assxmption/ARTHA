import yfinance as yf
ticker = yf.Ticker("RELIANCE.NS")
expirations = ticker.options
if expirations:
    opt = ticker.option_chain(expirations[0])
    print(opt.calls.head(1).to_dict('records'))
