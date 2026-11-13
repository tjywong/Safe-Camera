"""The lock's brain, with no camera / screen / GPIO code in it.

The same class runs on the laptop (lock.py) and on the UNO Q, so only the
input/output layer has to change when moving to hardware.

Only a LONG press followed by the right code opens the lock. After a SHORT
press everything looks the same, but the attempt always ends as wrong, even
with the right code. MAX_WRONG wrong attempts in a row (short-press ones
included) lock the button out for LOCKOUT_SECONDS.

Flow:
  IDLE --button--> ENTERING (both LEDs flash)
     for each of 3 digits:
       hold N fingers steadily (STABLE_FRAMES frames, prob >= CONFIDENCE) -> digit recorded
       then show a fist / remove your hand -> ready for the next digit
       (this is what lets the same number appear twice and avoids double counting)
  after 3 digits -> UNLOCKED if long press and right code, otherwise DENIED
                    (or LOCKED_OUT on the MAX_WRONG-th wrong attempt in a row)
  ENTERING times out to DENIED (a wrong attempt).
  UNLOCKED stays open until the button is pressed (-> IDLE, locked).
  DENIED stays until the button is pressed (-> ENTERING again).
  LOCKED_OUT ignores the button, then returns to IDLE with the count reset.
  Pressing the button while ENTERING starts the code over.
"""
from enum import Enum

import config


class State(Enum):
    IDLE = "idle"
    ENTERING = "entering"
    UNLOCKED = "unlocked"
    DENIED = "denied"
    LOCKED_OUT = "locked_out"


class Lock:
    def __init__(self, combination=config.COMBINATION, max_wrong=config.MAX_WRONG,
                 lockout_seconds=config.LOCKOUT_SECONDS, stable_frames=config.STABLE_FRAMES,
                 confidence=config.CONFIDENCE, timeout=config.ENTRY_TIMEOUT):
        assert len(combination) == 3 and all(1 <= d <= 5 for d in combination)
        self.combination = tuple(combination)
        self.max_wrong, self.lockout_seconds = max_wrong, lockout_seconds
        self.stable_frames, self.confidence = stable_frames, confidence
        self.timeout = timeout
        self.state, self.since = State.IDLE, 0.0
        self.wrong = 0                 # wrong attempts in a row
        self.locked_until = 0.0
        self.genuine = False           # this attempt started with a long press
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

    def lock_out(self, until, now):
        """Enter (or, after a restart, resume) the lockout until `until`."""
        self.locked_until = until
        self._go(State.LOCKED_OUT, now)

    def press_button(self, now, long_press=True):
        """Returns "close" if this press locked an open lock, "ignored" during a
        lockout, otherwise "enter"."""
        if self.state == State.UNLOCKED:
            self._go(State.IDLE, now)
            return "close"
        if self.state == State.LOCKED_OUT:
            return "ignored"
        self._reset_entry()            # IDLE, DENIED, or restarting mid-entry
        self.genuine = long_press
        self._go(State.ENTERING, now)
        return "enter"

    def _wrong_attempt(self, now):
        self.wrong += 1
        if self.wrong >= self.max_wrong:
            self.lock_out(now + self.lockout_seconds, now)
            return "lockout"
        self._go(State.DENIED, now)
        return None

    def update(self, count, prob, now):
        """Feed one prediction (fingers 0-5, softmax prob). Returns an event string or None."""
        if self.state == State.LOCKED_OUT and now >= self.locked_until:
            self.wrong = 0
            self._go(State.IDLE, now)
            return "lockout_over"
        if self.state != State.ENTERING:
            return None
        if now - self.since > self.timeout:
            return self._wrong_attempt(now) or "timeout"

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
        if self.genuine and tuple(self.entered) == self.combination:
            self.wrong = 0
            self._go(State.UNLOCKED, now)
            return "unlock"
        return self._wrong_attempt(now) or "deny"


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

    def attempt(lock, code, t, long_press=True):
        lock.press_button(t, long_press)
        return feed(lock, [code[0], 0, code[1], 0, code[2]], t)

    CODE, WRONG = (3, 1, 4), (1, 1, 1)

    lk = Lock(CODE)
    lk.press_button(0.0)
    ev, t = feed(lk, [3, 0, 1, 0, 4], 0.0)
    assert lk.state == State.UNLOCKED, (lk.state, ev)
    lk = Lock(CODE)
    lk.press_button(0.0)
    feed(lk, [3, 0, 2, 0, 4], 0.0)
    assert lk.state == State.DENIED
    lk = Lock((2, 2, 5))
    lk.press_button(0.0)
    feed(lk, [2, 2, 2, 0, 2, 0, 5], 0.0)        # holding 2 longer must NOT count twice
    assert lk.state == State.UNLOCKED, lk.entered
    lk = Lock(CODE)
    lk.update(3, 0.99, 0.0)
    assert lk.state == State.IDLE              # camera ignored until button pressed
    lk.press_button(0.0)
    assert lk.update(3, 0.99, 25.0) == "timeout"
    lk = Lock(CODE)
    lk.press_button(0.0)
    ev, t = feed(lk, [3, 0, 1, 0, 4], 0.0)
    lk.update(None, 0.0, t + 600)
    assert lk.state == State.UNLOCKED          # stays open until the button
    assert lk.press_button(t + 601) == "close" and lk.state == State.IDLE
    lk.press_button(0.0)
    ev, t = feed(lk, [3, 0, 2, 0, 4], 0.0)
    lk.update(None, 0.0, t + 600)
    assert lk.state == State.DENIED            # stays denied until the button
    assert lk.press_button(t + 601) == "enter" and lk.state == State.ENTERING
    feed(lk, [3], t + 601)
    lk.press_button(t + 700)
    assert lk.state == State.ENTERING and lk.entered == []   # press mid-entry restarts

    # Short press: the right code still ends as wrong, and counts as wrong.
    lk = Lock(CODE)
    ev, t = attempt(lk, CODE, 0.0, long_press=False)
    assert lk.state == State.DENIED and lk.wrong == 1 and "deny" in ev
    attempt(lk, CODE, t, long_press=True)
    assert lk.state == State.UNLOCKED and lk.wrong == 0

    # 3 wrong in a row -> 30 s lockout that ignores the button, then a fresh start.
    lk = Lock(CODE, lockout_seconds=30)
    attempt(lk, WRONG, 0.0)
    attempt(lk, CODE, 0.0, long_press=False)   # short-press attempts count too
    ev, t = attempt(lk, WRONG, 0.0)
    assert lk.state == State.LOCKED_OUT and "lockout" in ev
    assert lk.press_button(t + 1) == "ignored" and lk.state == State.LOCKED_OUT
    assert lk.update(None, 0.0, t + 29) is None and lk.state == State.LOCKED_OUT
    assert lk.update(None, 0.0, t + 30) == "lockout_over"
    assert lk.state == State.IDLE and lk.wrong == 0
    attempt(lk, CODE, t + 31)
    assert lk.state == State.UNLOCKED

    # A right code resets the count, so wrong attempts must be in a row.
    lk = Lock(CODE)
    attempt(lk, WRONG, 0.0)
    attempt(lk, WRONG, 0.0)
    attempt(lk, CODE, 0.0)
    lk.press_button(0.0)
    attempt(lk, WRONG, 0.0)
    assert lk.state == State.DENIED and lk.wrong == 1
    print("lock_logic self-test passed")
