import tushare as ts


def get_tushare_client(token: str):
    ts.set_token(token)
    return ts.pro_api()
