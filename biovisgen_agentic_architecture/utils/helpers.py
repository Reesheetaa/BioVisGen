import os
from datetime import datetime


def create_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_filename(text):
    return "".join(c for c in text if c.isalnum() or c in " _-")[:50]