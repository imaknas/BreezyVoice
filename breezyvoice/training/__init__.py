"""Adapting the text-to-speech-token language model to one speaker's recordings.

Only the language model is trained: it decides wording, rhythm and pronunciation, which is where a
voice cloned from a single reference clip goes wrong from take to take. The flow decoder and the
vocoder are used as released.
"""
