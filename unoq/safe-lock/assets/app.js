const ui = new WebUI();
const $ = (s) => document.querySelector(s);

ui.on_connect(() => { $('#conn').textContent = 'connected'; $('#conn').className = 'pill on'; });
ui.on_disconnect(() => { $('#conn').textContent = 'disconnected'; $('#conn').className = 'pill off'; });

// One row per class: 0 (fist / no hand) .. 5 fingers
$('#bars').innerHTML = [0, 1, 2, 3, 4, 5]
  .map((n) => `<div class="row"><span>${n}</span><div class="track"><div class="fill"></div><div class="thr"></div></div><b>–</b></div>`)
  .join('');
const rows = [...document.querySelectorAll('#bars .row')];

function message(v) {
  switch (v.state) {
    case 'entering': {
      const t = v.time_left == null ? '' : ` · ${Math.ceil(v.time_left)} s left`;
      return v.need_clear
        ? `Digit ${v.digits} accepted. Show a fist or move your hand away${t}`
        : `Show digit ${v.digits + 1} of 3${t}`;
    }
    case 'unlocked': return 'Unlocked. Press the button to lock';
    case 'denied': return 'Wrong code. Press the button to try again';
    case 'locked_out': return `Too many wrong attempts. Try again in ${Math.ceil(v.lockout_left)} s`;
    default: return 'Locked. Press the button to enter the code';
  }
}

function attempts(v) {
  if (v.state === 'locked_out' || !v.wrong) return '';
  return `Wrong attempts in a row: ${v.wrong} of ${v.max_wrong}`;
}

ui.on_message('view', (v) => {
  $('#cam').src = v.image;
  $('#nocam').hidden = true;

  $('#banner').textContent = message(v);
  $('#banner').className = `banner ${v.state === 'locked_out' ? 'denied' : v.state}`;
  $('#attempts').textContent = attempts(v);
  $('#attempts').hidden = !attempts(v);

  const best = v.probs.indexOf(Math.max(...v.probs));
  const p = v.probs[best];
  const sure = p >= v.confidence;
  $('#count').textContent = best;
  $('#conf').textContent = `${Math.round(p * 100)}%`;
  $('#conf').className = `big ${sure ? 'ok' : 'low'}`;
  $('#thr').textContent = `${Math.round(v.confidence * 100)}%`;
  $('#hold').style.width = `${Math.min(1, v.hold) * 100}%`;

  document.querySelectorAll('#dots span').forEach((d, i) => d.classList.toggle('on', i < v.digits));

  rows.forEach((r, i) => {
    const fill = r.querySelector('.fill');
    fill.style.width = `${v.probs[i] * 100}%`;
    fill.classList.toggle('best', i === best);
    fill.classList.toggle('sure', i === best && sure);
    r.querySelector('.thr').style.left = `${v.confidence * 100}%`;
    r.querySelector('b').textContent = `${Math.round(v.probs[i] * 100)}%`;
  });
});
