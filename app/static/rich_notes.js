// Shared notes that remember who wrote what (per-person projects;
// partials/_card_notes.html). The text is a list of runs [[member id or null, text, day or null], ...].
// Each run is drawn as a span in its author's color, and hovering it shows the name and the day it was
// written (runs from before 2026-11-04 have no day). On every edit, the changed stretch (between the
// unchanged start and the unchanged end) gets the person typing and today's date. Saved 1.2 s after the
// last keystroke, and on leaving the field. Like the plain notes (base.html), unsaved text is kept in
// localStorage until the server has it, and goes out again once the connection is back. If the note
// changed elsewhere meanwhile, what you wrote is added below the saved version.
(function () {
    function people(el) {
        var raw = JSON.parse(el.parentElement.querySelector('.rich-notes-people').textContent);
        var out = {};
        raw.people.forEach(function (id, i) { out[id] = { name: raw.names[i], color: raw.colors[i] }; });
        return out;
    }

    function textOf(el) {
        // Plain text as the browser shows it: text nodes, plus a newline for <br> and block elements
        // (some browsers insert those for Enter, even in plaintext-only mode)
        var out = '';
        (function walk(node) {
            node.childNodes.forEach(function (c) {
                if (c.nodeType === 3) out += c.nodeValue;
                else if (c.nodeName === 'BR') out += '\n';
                else {
                    if (c.nodeName === 'DIV' && out && !out.endsWith('\n')) out += '\n';
                    walk(c);
                }
            });
        })(el);
        return out;
    }

    // Who wrote each character, and when: [member id, day]
    function authorsOf(runs) {
        var a = [];
        runs.forEach(function (r) { for (var i = 0; i < r[1].length; i++) a.push([r[0], r[2] || null]); });
        return a;
    }

    function runsOf(text, authors) {
        var runs = [];
        for (var i = 0; i < text.length; i++) {
            var last = runs[runs.length - 1], a = authors[i];
            if (last && last[0] === a[0] && last[2] === a[1]) last[1] += text[i];
            else runs.push([a[0], text[i], a[1]]);
        }
        return runs;
    }

    function today() {
        var d = new Date(), pad = function (n) { return (n < 10 ? '0' : '') + n; };
        return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate());
    }

    function dayLabel(day) {  // "2026-09-28" -> "Sep 28, 2026"
        return new Date(day + 'T12:00:00').toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    function caretOffset(el) {
        var sel = window.getSelection();
        if (!sel.rangeCount || !el.contains(sel.anchorNode)) return null;
        var r = sel.getRangeAt(0).cloneRange();
        r.setStart(el, 0);
        var frag = document.createElement('div');
        frag.appendChild(r.cloneContents());
        return textOf(frag).length;
    }

    function setCaret(el, offset) {
        var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), node, left = offset;
        while ((node = walker.nextNode())) {
            if (left <= node.nodeValue.length) {
                var r = document.createRange();
                r.setStart(node, left);
                r.collapse(true);
                var sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(r);
                return;
            }
            left -= node.nodeValue.length;
        }
    }

    function draw(el) {
        var who = el._people;
        el.textContent = '';
        el._runs.forEach(function (r) {
            var span = document.createElement('span');
            span.textContent = r[1];
            var p = r[0] != null && who[r[0]];
            span.dataset.m = r[0] == null ? '' : r[0];
            span.dataset.tip = (p ? p.name : 'Written before notes were per person') + (r[2] ? ', ' + dayLabel(r[2]) : '');
            if (p) span.style.color = p.color;
            el.appendChild(span);
        });
    }

    function status(el, text, ok) {
        var s = el.parentElement.querySelector('.rich-notes-status');
        s.textContent = text;
        s.style.color = ok === null ? 'var(--ink-3)' : ok ? 'var(--success)' : 'var(--error)';
        s.style.display = text ? '' : 'none';
    }

    function plain(runs) { return runs.map(function (r) { return r[1]; }).join(''); }
    function draftKey(el) { return 'rich-notes-draft:' + el.dataset.arxivId + ':' + el.dataset.projectId; }
    function writeDraft(el) {
        localStorage.setItem(draftKey(el), JSON.stringify({ base: el._base, runs: el._runs }));
    }

    // Puts what changed between base and mine below the note as it is on the server now
    function mergeMine(el, base, mine, server) {
        var old = base, now = plain(mine), p = 0, s = 0;
        while (p < old.length && p < now.length && old[p] === now[p]) p++;
        while (s < old.length - p && s < now.length - p && old[old.length - 1 - s] === now[now.length - 1 - s]) s++;
        var added = now.slice(p, now.length - s), addedBy = authorsOf(mine).slice(p, now.length - s);
        var saved = plain(server), gap = saved && added ? '\n\n' : '';
        var me = [parseInt(el.dataset.me, 10), today()];
        el._runs = runsOf(saved + gap + added, authorsOf(server).concat(gap ? [me, me] : [], addedBy));
        el._base = saved;
        el._notice = 'changed elsewhere too, your text is added below';
        draw(el);
    }

    function save(el) {
        clearTimeout(el._timer);
        if (el._saving) return el._saving;
        var sent = plain(el._runs), answered = false;
        var body = new URLSearchParams({ project_id: el.dataset.projectId, runs: JSON.stringify(el._runs), base: el._base });
        status(el, 'saving…', null);
        el._saving = fetch('/paper/' + encodeURIComponent(el.dataset.arxivId) + '/notes-runs',
            { method: 'POST', body: body, signal: AbortSignal.timeout(30000) })
            .then(function (r) {
                answered = true;
                rwNet.answered();
                el._saving = null;
                if (r.status === 409) {
                    return r.json().then(function (d) {
                        mergeMine(el, el._base, el._runs, d.runs);
                        writeDraft(el);
                        return save(el);
                    });
                }
                if (!r.ok) throw new Error(r.status);
                el._base = sent;
                if (plain(el._runs) !== sent) return save(el);  // typed on while this was on its way
                el._dirty = false;
                localStorage.removeItem(draftKey(el));
                if (el._notice) { status(el, el._notice, false); return; }
                status(el, 'saved', true);
                setTimeout(function () { if (!el._dirty && !el._notice) status(el, '', true); }, 1500);
            })
            .catch(function () {
                el._saving = null;
                writeDraft(el);
                status(el, answered ? 'not saved, server error' : 'not saved, kept in this browser, will retry when online', false);
                if (!answered) rwNet.failed();
                rwNet.sync();
            });
        return el._saving;
    }

    function onInput(el) {
        if (el._composing) return;
        var old = el._runs.map(function (r) { return r[1]; }).join('');
        var now = textOf(el);
        var authors = authorsOf(el._runs);
        var p = 0;
        while (p < old.length && p < now.length && old[p] === now[p]) p++;
        var s = 0;
        while (s < old.length - p && s < now.length - p && old[old.length - 1 - s] === now[now.length - 1 - s]) s++;
        var me = parseInt(el.dataset.me, 10), day = today();
        var inserted = now.length - p - s;
        var mid = [];
        for (var i = 0; i < inserted; i++) mid.push([me, day]);
        el._runs = runsOf(now, authors.slice(0, p).concat(mid, authors.slice(old.length - s)));
        var caret = caretOffset(el);
        draw(el);
        if (caret !== null) setCaret(el, caret);
        el._dirty = true;
        el._notice = null;
        writeDraft(el);
        status(el, 'saving…', null);
        clearTimeout(el._timer);
        el._timer = setTimeout(function () { save(el); }, 1200);
    }

    function init(el) {
        if (el._runs) return;
        el._runs = JSON.parse(el.dataset.runs || '[]');
        el._base = plain(el._runs);
        el._people = people(el);
        el._noteDirty = function () { return !!el._dirty; };
        el._noteRefresh = function (d) {
            if (plain(d.runs) !== el._base) { el._runs = d.runs; el._base = plain(d.runs); draw(el); }
        };
        el._flushNote = function () { return el._saving || (el._dirty ? save(el) : Promise.resolve()); };
        var raw = localStorage.getItem(draftKey(el));
        if (raw !== null) {
            var draft = JSON.parse(raw);
            if (plain(draft.runs) === el._base) localStorage.removeItem(draftKey(el));
            else {
                if (draft.base === el._base) el._runs = draft.runs;
                else mergeMine(el, draft.base, draft.runs, el._runs);
                el._dirty = true;
                writeDraft(el);
            }
        }
        draw(el);
        if (el._dirty) {
            status(el, el._notice || 'not saved yet', false);
            if (navigator.onLine) setTimeout(function () { save(el); }, 0);
        }
        el.addEventListener('compositionstart', function () { el._composing = true; });
        el.addEventListener('compositionend', function () { el._composing = false; onInput(el); });
        el.addEventListener('input', function () { onInput(el); });
        el.addEventListener('blur', function () { if (el._dirty) save(el); });
    }

    function initAll(root) {
        (root || document).querySelectorAll('.rich-notes').forEach(init);
    }
    document.addEventListener('DOMContentLoaded', function () { initAll(); });
    document.body.addEventListener('htmx:afterSettle', function (e) { initAll(e.detail.elt); });
    if (window.unsavedChecks) {
        window.unsavedChecks.push(function () {
            return Array.prototype.some.call(document.querySelectorAll('.rich-notes'), function (el) { return el._dirty; });
        });
    }
    window.addEventListener('rw:back', function () {
        document.querySelectorAll('.rich-notes').forEach(function (el) { if (el._dirty) save(el); });
    });
})();
