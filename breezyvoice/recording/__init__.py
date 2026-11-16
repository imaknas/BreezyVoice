"""From a person reading a script aloud to one audio clip per script line.

A reader records a whole section in one file, repeats lines and restarts after slips. The recording
is cut at silences, each piece is recognised, and neighbouring pieces are grouped so that the groups
match script lines as well as possible.
"""
