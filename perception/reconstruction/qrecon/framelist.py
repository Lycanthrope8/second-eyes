"""Frame lists (rgb.txt) on the laptop and on the headset: whether the headset's copy is out of date."""


def normalised(text):
    return "\n".join(l.strip() for l in (text or "").splitlines() if l.strip())


def count(text):
    return sum(1 for l in (text or "").splitlines() if l.strip() and not l.startswith("#"))


def differs(local_text, remote_text):
    """True when the headset's list is not the laptop's. Comparing the whole list (its header records when the frames
    were extracted) catches a re-extraction that kept the same number of frames, which a count would miss."""
    return normalised(local_text) != normalised(remote_text)
