/* Progressive enhancement for explicitly supported, non-executed AIgile DIL exports.
 * Operates only on generated data-aigile-readiness markup. Nothing from the
 * source document is evaluated or used as a script or HTML template.
 */
(function () {
  'use strict';

  const interpretations = [
    'Foundational readiness is limited. Focus the initial effort on reliable engineering practices, governance, and a small controlled pilot rather than broad agent autonomy.',
    'There is a workable foundation for a structured pilot. Address the weaker areas before scaling across teams.',
    'The organization appears reasonably positioned for structured pilots and selective expansion, subject to evidence from actual delivery outcomes.'
  ];

  function install(root) {
    if (!root || root.dataset.aigileInstalled === '1') return;
    const areas = Array.from(root.querySelectorAll('[data-aigile-area]'));
    if (areas.length !== 7 || areas.some((area, index) => Number(area.dataset.aigileArea) !== index)) return;
    root.dataset.aigileInstalled = '1';
    const ratings = Array(7).fill(0);
    const total = root.querySelector('[data-aigile-total]');
    const totalLabel = root.querySelector('[data-aigile-total-label]');
    const interpretation = root.querySelector('[data-aigile-interpretation]');
    const priorities = root.querySelector('[data-aigile-priorities]');
    const request = root.querySelector('[data-aigile-request]');
    const prepare = root.querySelector('[data-aigile-prepare]');
    if (!total || !totalLabel || !interpretation || !priorities || !request || !prepare) return;

    function update() {
      const answered = ratings.filter(Boolean).length;
      const score = ratings.reduce((sum, rating) => sum + rating, 0);
      root.dataset.aigileComplete = answered === 7 ? 'true' : 'false';
      total.textContent = answered === 7 ? score + '/35' : answered + '/7';
      totalLabel.textContent = answered === 7 ? 'Overall score' : 'Areas rated';
      areas.forEach((area, index) => {
        area.querySelector('[data-aigile-area-score]').textContent = ratings[index] ? ratings[index] + ' / 5' : 'Not rated';
        area.querySelectorAll('[data-aigile-rate]').forEach(button => {
          const selected = Number(button.dataset.aigileRate) === ratings[index];
          button.setAttribute('aria-pressed', String(selected));
        });
      });
      if (answered !== 7) {
        priorities.replaceChildren();
        request.textContent = '';
        return;
      }
      interpretation.textContent = interpretations[score < 18 ? 0 : score < 28 ? 1 : 2];
      priorities.replaceChildren();
      ratings.map((rating, index) => ({ rating, index })).sort((a, b) => a.rating - b.rating || a.index - b.index)
        .slice(0, 3).forEach(item => {
          const entry = document.createElement('div');
          entry.className = 'aigile-priority';
          const name = document.createElement('span');
          name.textContent = areas[item.index].querySelector('.aigile-readiness-label strong').textContent;
          const value = document.createElement('strong');
          value.textContent = item.rating + '/5';
          entry.append(name, value);
          priorities.append(entry);
        });
    }

    root.addEventListener('click', event => {
      const button = event.target.closest('button[data-aigile-rate]');
      if (!button || !root.contains(button)) return;
      const index = Number(button.dataset.aigileIndex);
      const rating = Number(button.dataset.aigileRate);
      if (!Number.isInteger(index) || index < 0 || index >= 7 || !Number.isInteger(rating) || rating < 1 || rating > 5) return;
      ratings[index] = ratings[index] === rating ? 0 : rating;
      update();
    });
    prepare.addEventListener('click', () => {
      if (ratings.some(r => !r)) return;
      const report = areas.map((area, index) => area.querySelector('.aigile-readiness-label strong').textContent + ': ' + ratings[index] + '/5').join('; ');
      request.textContent = 'Using my AIgile organizational readiness self-assessment ratings - ' + report + ' - recommend my top strategic investments, organizational risks, and a tailored 90-day implementation plan.';
      request.focus();
      const selection = window.getSelection();
      if (selection) {
        const range = document.createRange();
        range.selectNodeContents(request);
        selection.removeAllRanges();
        selection.addRange(range);
      }
    });
    update();
  }

  function installAll(container) {
    if (!container || !container.querySelectorAll) return;
    container.querySelectorAll('[data-aigile-readiness="1"]').forEach(install);
  }

  window.MdAigileWidgets = { install: installAll };
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => installAll(document), { once: true });
  } else {
    installAll(document);
  }
})();
