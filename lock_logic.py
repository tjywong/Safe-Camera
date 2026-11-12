"""The lock's brain, with no camera / screen / GPIO code in it.

The same class runs on the laptop (lock.py) and on the UNO Q, so only the
input/output layer has to change when moving to hardware.

Flow:
  IDLE --button--> ENTERING (LEDs on)
     for each of 3 digits:
       hold N fingers steadily (STABLE_FRAMES frames, prob >= CONFIDENCE) -> digit recorded
       then show a fist / remove your hand -> ready for the next digit
       (this is what lets the same number appear twice and avoids double counting)
  after 3 digits -> UNLOCKED if they match the combination, otherwise DENIED
  ENTERING times out to DENIED; UNLOCKED / DENIED return to IDLE after a few seconds.
"""
from enum import Enum

import config


class State(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    UNLOCKED = "unlocked"
    DENIED = "denied"


class Lock:
    def __init__(self, combination=config.COMBINATION, stable_frames=config.STABLE_FRAMES,
                 confidence=config.CONFIDENCE, timeout=config.ENTRY_TIMEOUT,
                 unlock_seconds=config.UNLOCK_SECONDS, denied_seconds=3.0):
        assert len(combination) == 3 and all(1 <= d <= 5 for d in combination)
        self.combination = tuple(combination)
        self.stable_frames, self.confidence = stable_frames, confidence
        self.timeout, self.unlock_seconds, self.denied_seconds = timeout, unlock_seconds, denied_seconds
        self.state, self.since = State.IDLE, 0.0
        self._reset_entry()

    def _reset_entry(self):
        self.entered = []
        self.need_clear = False
        self.streak_value, self.streak = None, 0

    def _go(self, state, now):
        self.state, self.since = state, now

    @property
    def leds_on(self):
        return self.state == State.ENTERING

    def press_button(self, now):
        if self.state in (State.IDLE, State.DENIED):
            self._reset_entry()
            self._go(State.ENTERING, now)

    def update(self, count, prob, now):
        """Feed one prediction (fingers 0-5, softmax prob). Returns an event string or None."""
        if self.state == State.UNLOCKED and now - self.since > self.unlock_seconds:
            self._go(State.IDLE, now)
            return "relock"
        if self.state == State.DENIED and now - self.since > self.denied_seconds:
            self._go(State.IDLE, now)
            return None
        if self.state != State.ENTERING:
            return None
        if now - self.since > self.timeout:
            self._go(State.DENIED, now)
            return "timeout"

        if count is None or prob < self.confidence:
            self.streak_value, self.streak = None, 0
            return None
        if count == self.streak_value:
            self.streak += 1
        else:
            self.streak_value, self.streak = count, 1
        if self.streak != self.stable_frames:   # fire exactly once per steady pose
            return None

        if self.need_clear:
            if count == 0:
                self.need_clear = False
                return "cleared"
            return None
        if count == 0:
            return None
        self.entered.append(count)
        self.need_clear = True
        if len(self.entered) < 3:
            return "digit"
        ok = tuple(self.entered) == self.combination
        self._go(State.UNLOCKED if ok else State.DENIED, now)
        return "unlock" if ok else "deny"


if __name__ == "__main__":
    # Quick self-test with fake predictions.
    def feed(lock, seq, t):
        events = []
        for c in seq:
            for _ in range(config.STABLE_FRAMES):
                t += 0.03
                e = lock.update(c, 0.99, t)
                if e:
                    events.append(e)
        return events, t

    lk = Lock((3, 1, 4))
    lk.press_button(0.0)
    ev, t = feed(lk, [3, 0, 1, 0, 4], 0.0)
    assert lk.state == State.UNLOCKED, (lk.state, ev)
    lk = Lock((3, 1, 4))
    lk.press_button(0.0)
    feed(lk, [3, 0, 2, 0, 4], 0.0)
    assert lk.state == State.DENIED
    lk = Lock((2, 2, 5))
    lk.press_button(0.0)
    feed(lk, [2, 2, 2, 0, 2, 0, 5], 0.0)        # holding 2 longer must NOT count twice
    assert lk.state == State.UNLOCKED, lk.entered
    lk = Lock((3, 1, 4))
    lk.update(3, 0.99, 0.0)
    assert lk.state == State.IDLE              # camera ignored until button pressed
    lk.press_button(0.0)
    assert lk.update(3, 0.99, 25.0) == "timeout"
    print("lock_logic self-test passed")
