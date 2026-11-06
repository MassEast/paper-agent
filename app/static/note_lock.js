// One writer at a time in a note. The field you are in holds a lock on the server (app/note_locks.py), renewed
// every 10 s while you were active in the last minute. The note shown on screen asks every 5 s whether someone
// else is writing (the field then goes read-only and says who) and picks up what they wrote.
// Works on both note editors through _noteDirty and _noteRefresh, which each sets on its field.
(function () {
    var FIELDS = 'textarea[data-arxiv-id], .rich-notes';
    var client = Math.random().toString(36).slice(2);
    var active = null, lastUsed = 0;

    function url(el, path) { return '/paper/' + encodeURIComponent(el.dataset.arxivId) + path; }
    function form(el, extra) {
        return new URLSearchParams(Object.assign({ project_id: el.dataset.projectId, client: client }, extra));
    }

    function setLocked(el, by) {  // by: a name, "" for someone unnamed, null for nobody
        var locked = by !== null && !el._noteDirty();
        if (locked === !!el._lockedShown) return;
        el._lockedShown = locked;
        if (el.tagName === 'TEXTAREA') el.readOnly = locked;
        else el.contentEditable = locked ? 'false' : 'plaintext-only';
        el.style.opacity = locked ? '0.7' : '';
        var status = el.parentElement.querySelector('.rich-notes-status, [id^="notes-saved-"]');
        status.textContent = locked ? (by || 'Someone else') + ' is writing here' : '';
        status.style.color = 'var(--ink-3)';
        status.style.display = locked ? '' : 'none';
    }

    function claim(el) {
        fetch(url(el, '/note-lock'), { method: 'POST', body: form(el) })
            .then(function (r) { return r.json(); })
            .then(function (d) { setLocked(el, d.by); })
            .catch(function () {});  // offline: keep writing, the save checks for changes later
    }

    function release(el) {
        navigator.sendBeacon(url(el, '/note-lock'), form(el, { release: 1 }));
    }

    function look() {
        if (document.hidden) return;
        document.querySelectorAll(FIELDS).forEach(function (el) {
            if ((el === active && !el._lockedShown) || !el.offsetParent) return;
            fetch(url(el, '/note-state') + '?' + form(el))
                .then(function (r) { return r.json(); })
                .then(function (d) {
                    setLocked(el, d.by);
                    if (d.by === null && !el._noteDirty()) el._noteRefresh(d);
                    if (d.by === null && el === active) claim(el);  // clicked in while it was locked
                })
                .catch(function () {});  // the banner reports a lost connection
        });
    }

    document.addEventListener('focusin', function (e) {
        if (!e.target.matches(FIELDS)) return;
        active = e.target;
        lastUsed = Date.now();
        claim(active);
    });
    document.addEventListener('input', function (e) {
        if (e.target.matches && e.target.matches(FIELDS)) lastUsed = Date.now();
    });
    document.addEventListener('focusout', function (e) {
        if (e.target !== active) return;
        active = null;
        release(e.target);
    });
    window.addEventListener('pagehide', function () { if (active) release(active); });
    setInterval(function () {
        if (active && Date.now() - lastUsed < 60000) claim(active);
    }, 10000);
    setInterval(look, 5000);
})();
