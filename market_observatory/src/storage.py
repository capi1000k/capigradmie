# src/storage.py — v2 stub (DuckDB ga ko'chgandik, bu faqat import uchun)
from src.db import insert_klines, insert_trades, insert_orderbook
from src.db import load_klines, load_trades, load_orderbook

def save_klines(df, interval):
    insert_klines(df, interval)

def save_trades(df):
    insert_trades(df)

def save_orderbook(df):
    insert_orderbook(df)
