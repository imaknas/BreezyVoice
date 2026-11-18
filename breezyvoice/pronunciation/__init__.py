"""Checking that a take was pronounced as intended, tone and reading included.

A recogniser that writes characters cannot do this: it chooses the character from context, so 調 read
as diào still comes out as 调. Here the audio is scored against sound units directly (initials and
finals with their tone), and for every character the intended reading competes with the readings it
could be mistaken for. The intended reading comes from the same predictor the synthesizer uses.
"""
