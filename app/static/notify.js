// The Notify box on a card (paper_card.html): an optional message, then the email to the project's addresses.
// The note is saved first so the email has all of it. A failed send keeps the box and the message.
(function () {
    function boxOf(el) { return el.closest('.paper-card').querySelector('.notify-box'); }

    function say(box, text, ok) {
        var s = box.querySelector('.notify-status');
        s.textContent = text;
        s.style.color = ok ? 'var(--success)' : 'var(--error)';
    }

    window.openNotify = function (btn) {
        btn.closest('details').open = false;
        var box = boxOf(btn);
        box.hidden = false;
        say(box, '', true);
        box.querySelector('textarea').focus();
    };

    window.closeNotify = function (btn) { boxOf(btn).hidden = true; };

    window.sendNotify = function (btn) {
        var box = boxOf(btn), input = box.querySelector('textarea');
        btn.disabled = true;
        say(box, 'Sending…', true);
        flushNotes(box.closest('.paper-card')).then(function () {
            return fetch('/paper/' + encodeURIComponent(box.dataset.arxivId) + '/notify', {
                method: 'POST',
                body: new URLSearchParams({ project_id: box.dataset.projectId, message: input.value }),
                signal: AbortSignal.timeout(60000)
            });
        }).then(function (r) { return r.json(); }).then(function (d) {
            btn.disabled = false;
            if (!d.ok) { say(box, d.error || 'Send failed', false); return; }
            input.value = '';
            say(box, '✓ Sent', true);
            setTimeout(function () { box.hidden = true; }, 2000);
        }).catch(function () {
            btn.disabled = false;
            rwNet.failed();
            say(box, navigator.onLine ? 'No answer. It may have gone out, so check before sending again.'
                                      : 'No connection, not sent. Your message is still here.', false);
        });
    };
})();
