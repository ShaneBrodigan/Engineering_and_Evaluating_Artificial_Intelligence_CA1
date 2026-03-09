import spacy
import subprocess
import sys

class Config:
    ESSENTIAL_COLS = ["interaction_content", "ticket_summary"]

    def __init__(self):
        self.essential_cols = self.ESSENTIAL_COLS

        try:
            spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])