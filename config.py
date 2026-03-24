import spacy
import subprocess
import sys

class Config:
    ESSENTIAL_COLS = ["interaction_content", "ticket_summary"]
    MODEL_TARGET_COLS = ["type_2", "type_3", "type_4"]
    SELECTED_F1_AVERAGE = "macro"

    def __init__(self):
        self.essential_cols = self.ESSENTIAL_COLS
        self.model_target_cols = self.MODEL_TARGET_COLS

        try:
            spacy.load("en_core_web_sm")
        except OSError:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])