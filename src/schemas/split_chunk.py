
class SplitChunk:
    def __init__(self, text: str, start_index: int, end_index: int):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index

    def __repr__(self):
        return f"SplitChunk(start={self.start_index}, end={self.end_index})"