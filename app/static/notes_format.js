// Notes formatting (2026-11-12): while a note isn't being edited it is shown formatted: links are clickable, a
// link to a paper in this app (…/projects/<slug>?paper=<id> on this host) shows as that paper's title, and
// **text** is bold. Editing shows the plain text again (where bold is **text**); Ctrl/⌘+B or the B button next
// to "Notes" puts ** around the selection or takes them off. The stored note stays plain text.
// Every shown piece remembers where it starts in the plain text (data-s), so a click into a formatted note puts
// the cursor at the same place in the plain text. Used by the plain notes (base.html) and the per-person
// notes (rich_notes.js).
(function () {
    var TOKEN = /\*\*([^*\n]+?)\*\*|(https?:\/\/[^\s<>"']+)/;  // a fresh /g copy per render: render recurses
    var titles = {};  // paper id -> title (null while being fetched)
    var waiting = [];  // re-render callbacks for when titles arrive

    function trimUrl(url) {  // a sentence's full stop, or a closing bracket the link didn't open, isn't part of it
        url = url.replace(/[.,;:!?]+$/, '');
        while (/[)\]]$/.test(url)) {
            var close = url.slice(-1), open = close === ')' ? '(' : '[';
            if (url.split(open).length >= url.split(close).length) break;
            url = url.slice(0, -1).replace(/[.,;:!?]+$/, '');
        }
        return url;
    }

    function paperId(url) {
        try {
            var u = new URL(url);
            if (u.host !== window.location.host || !/^\/projects\/[\w-]+\/?$/.test(u.pathname)) return null;
            var id = u.searchParams.get('paper');
            return id && /^\d+$/.test(id) ? id : null;
        } catch (e) { return null; }
    }

    function fetchTitles(ids, rerender) {
        if (rerender && waiting.indexOf(rerender) < 0) waiting.push(rerender);
        var missing = ids.filter(function (id) { return !(id in titles); });
        if (!missing.length) return;
        missing.forEach(function (id) { titles[id] = null; });
        fetch('/papers/titles?ids=' + missing.join(','))
            .then(function (r) { return r.ok ? r.json() : {}; })
            .then(function (d) {
                missing.forEach(function (id) { titles[id] = d[id] || ''; });
                var cbs = waiting; waiting = [];
                cbs.forEach(function (cb) { cb(); });
            })
            .catch(function () { missing.forEach(function (id) { delete titles[id]; }); });
    }

    function piece(tag, text, start) {
        var el = document.createElement(tag);
        el.textContent = text;
        el.dataset.s = start;
        return el;
    }

    function link(url, start, rerender) {
        var a = document.createElement('a');
        var id = paperId(url);
        a.href = url;
        a.dataset.s = start;
        a.dataset.e = start + url.length;
        a.className = 'note-link';
        if (id) {
            var t = titles[id];
            if (!(id in titles)) fetchTitles([id], rerender);
            else if (t === null && rerender && waiting.indexOf(rerender) < 0) waiting.push(rerender);
            a.textContent = t ? t : (t === '' ? url : 'paper ' + id + '…');
            a.classList.add('note-paper-link');
            a.dataset.tip = t ? 'Paper in this app: ' + t : url;
        } else {
            a.textContent = url;
            a.target = '_blank';
            a.rel = 'noopener';
        }
        return a;
    }

    // Formatted nodes for text that starts at offset base in the whole note
    function render(text, base, rerender) {
        var out = [], last = 0, m, token = new RegExp(TOKEN.source, 'g');
        while ((m = token.exec(text))) {
            var start = m.index, raw = m[0];
            if (m[2]) {
                raw = trimUrl(m[2]);
                token.lastIndex = start + raw.length;
            }
            if (start > last) out.push(piece('span', text.slice(last, start), base + last));
            if (m[1] !== undefined) {
                var b = document.createElement('strong');
                b.dataset.s = base + start + 2;
                render(m[1], base + start + 2, rerender).forEach(function (n) { b.appendChild(n); });
                out.push(b);
            } else {
                out.push(link(raw, base + start, rerender));
            }
            last = start + raw.length;
        }
        if (last < text.length) out.push(piece('span', text.slice(last), base + last));
        return out;
    }

    // Where in the plain text a click at (x, y) inside a formatted note is
    function offsetAt(x, y, fallback) {
        var node, off;
        if (document.caretPositionFromPoint) {
            var p = document.caretPositionFromPoint(x, y);
            if (p) { node = p.offsetNode; off = p.offset; }
        } else if (document.caretRangeFromPoint) {
            var r = document.caretRangeFromPoint(x, y);
            if (r) { node = r.startContainer; off = r.startOffset; }
        }
        if (!node) return fallback;
        var el = node.nodeType === 3 ? node.parentElement : node;
        if (!el || el.dataset.s === undefined) return fallback;
        if (el.dataset.e !== undefined) return parseInt(el.dataset.e, 10);
        return parseInt(el.dataset.s, 10) + (node.nodeType === 3 ? off : 0);
    }

    // ** around [s, e) of text, or off again if they are there: {text, s, e} for the result and its selection
    function toggleBold(text, s, e) {
        if (text.slice(s - 2, s) === '**' && text.slice(e, e + 2) === '**')
            return { from: s - 2, to: e + 2, insert: text.slice(s, e), s: s - 2, e: e - 2 };
        var sel = text.slice(s, e);
        if (sel.length > 4 && sel.slice(0, 2) === '**' && sel.slice(-2) === '**')
            return { from: s, to: e, insert: sel.slice(2, -2), s: s, e: e - 4 };
        return { from: s, to: e, insert: '**' + sel + '**', s: s + 2, e: e + 2 };
    }

    // Replace [from, to) of a textarea with insert, keeping the browser's undo where it can
    function replaceInTextarea(t, from, to, insert) {
        t.focus();
        t.setSelectionRange(from, to);
        if (!document.execCommand || !document.execCommand('insertText', false, insert)) {
            t.setRangeText(insert, from, to, 'end');
            t.dispatchEvent(new Event('input', { bubbles: true }));
        }
    }

    // --- Plain notes (a textarea): a formatted view in front of it while it isn't focused ---
    function viewOf(t) { return t.parentElement.querySelector('.note-view'); }

    function showView(t) {
        var v = viewOf(t);
        if (!v) return;
        if (!t.value.trim() || document.activeElement === t) {
            v.hidden = true;
            t.style.display = '';
            return;
        }
        v.textContent = '';
        render(t.value, 0, function () { showView(t); }).forEach(function (n) { v.appendChild(n); });
        v.style.minHeight = t.offsetHeight ? t.offsetHeight + 'px' : '';
        v.hidden = false;
        t.style.display = 'none';
    }

    function edit(t, offset) {
        var v = viewOf(t);
        if (v) v.hidden = true;
        t.style.display = '';
        t.focus();
        var o = offset == null ? t.value.length : offset;
        t.setSelectionRange(o, o);
    }

    function attach(t) {
        if (t._noteView) { showView(t); return; }
        t._noteView = true;
        var v = document.createElement('div');
        v.className = 'note-view';
        v.hidden = true;
        v.tabIndex = 0;
        v.setAttribute('role', 'textbox');
        v.setAttribute('aria-label', 'Notes (click to edit)');
        t.parentElement.insertBefore(v, t);
        v.addEventListener('mousedown', function (e) {
            if (e.button !== 0) return;
            var a = e.target.closest('a');
            if (a) return;  // a link: let the click open it
            e.preventDefault();
            edit(t, offsetAt(e.clientX, e.clientY, t.value.length));
        });
        v.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && !e.target.closest('a')) { e.preventDefault(); edit(t); }
        });
        t.addEventListener('blur', function () { setTimeout(function () { showView(t); }, 0); });
        showView(t);
    }

    function boldTextarea(t) {
        var r = toggleBold(t.value, t.selectionStart, t.selectionEnd);
        replaceInTextarea(t, r.from, r.to, r.insert);
        t.setSelectionRange(r.s, r.e);
        t.dispatchEvent(new KeyboardEvent('keyup', { bubbles: true }));  // htmx saves on keyup
    }

    // The B button next to "Notes" (keeps the editor's focus and selection: mousedown is prevented)
    function boldButton(btn) {
        var card = btn.closest('.paper-card');
        var t = card && card.querySelector('textarea[data-arxiv-id]');
        var rich = card && card.querySelector('.rich-notes');
        if (t) {
            if (document.activeElement !== t) edit(t);
            boldTextarea(t);
        } else if (rich && rich._bold) {
            rich._bold();
        }
    }

    document.addEventListener('keydown', function (e) {
        if (!(e.metaKey || e.ctrlKey) || e.altKey || e.shiftKey || (e.key !== 'b' && e.key !== 'B')) return;
        var t = e.target;
        if (t.matches && t.matches('textarea[data-arxiv-id]')) { e.preventDefault(); boldTextarea(t); }
        else if (t.closest && t.closest('.rich-notes') && t.closest('.rich-notes')._bold) {
            e.preventDefault();
            t.closest('.rich-notes')._bold();
        }
    });

    window.noteFormat = { render: render, offsetAt: offsetAt, toggleBold: toggleBold, attach: attach,
                          refresh: showView, boldButton: boldButton, _titles: titles };
})();
