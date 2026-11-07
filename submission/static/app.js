const depths = ['24,1', '25,3', '26,5', '27,7'];
const files = Array(4).fill(null);
const previews = Array(4).fill(null);
const downloads = [];
const byId = (id) => document.getElementById(id);
const slots = byId('slots');
const buttons = [...document.querySelectorAll('button')];
const inputs = [];
const pngURL = (s) => 'data:image/png;base64,' + s;

function change(i, file) {
  if (previews[i]) URL.revokeObjectURL(previews[i]);
  files[i] = file;
  previews[i] = file ? URL.createObjectURL(file) : null;
  const preview = byId('preview-' + i);
  preview.replaceChildren();
  if (file) {
    const img = document.createElement('img');
    img.src = previews[i];
    img.alt = 'Цель для плоскости ' + (i + 1);
    preview.append(img);
  } else preview.textContent = 'PNG не выбран';
  byId('results').hidden = true;
  byId('error').hidden = true;
  byId('status').textContent = '';
}

function lock(busy) {
  [...buttons, ...inputs].forEach(el => el.disabled = busy);
}

function error(message) {
  byId('error').textContent = message;
  byId('error').hidden = false;
  byId('status').textContent = '';
}

depths.forEach((depth, i) => {
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = '<label for="file-' + i + '">Плоскость ' + (i + 1) + ' · ' + depth +
    ' мм</label><input type="file" id="file-' + i + '" accept="image/png,.png">' +
    '<div class="preview" id="preview-' + i + '">PNG не выбран</div>';
  slots.append(card);
  const input = byId('file-' + i);
  inputs.push(input);
  input.addEventListener('change', () => change(i, input.files[0] || null));
});

document.querySelectorAll('[data-example]').forEach(button => {
  button.addEventListener('click', async () => {
    lock(true);
    try {
      const sample = await Promise.all(depths.map(async (_, i) => {
        const res = await fetch('/examples/' + button.dataset.example + '/' + (i + 1));
        if (!res.ok) throw new Error('Не удалось загрузить пример.');
        return new File([await res.blob()], 'plane_' + (i + 1) + '.png', {type:'image/png'});
      }));
      sample.forEach((file, i) => { inputs[i].value = ''; change(i, file); });
      byId('status').textContent = 'Пример загружен. Можно синтезировать маску.';
    } catch (e) { error(e.message); }
    finally { lock(false); }
  });
});

function download(id, encoded, type) {
  const data = Uint8Array.from(atob(encoded), c => c.charCodeAt(0));
  const url = URL.createObjectURL(new Blob([data], {type}));
  downloads.push(url);
  byId(id).href = url;
}

byId('form').addEventListener('submit', async event => {
  event.preventDefault();
  byId('error').hidden = true;
  byId('results').hidden = true;
  while (downloads.length) URL.revokeObjectURL(downloads.pop());
  if (files.some(file => !file)) { error('Выберите четыре PNG, по одному на плоскость.'); return; }
  const form = new FormData();
  files.forEach((file, i) => form.append('plane_' + (i + 1), file));
  byId('status').textContent = 'Выполняется синтез и проверка маски…';
  lock(true);
  try {
    const res = await fetch('/api/synthesize', {method:'POST', body:form});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Не удалось выполнить синтез.');
    byId('mask').src = pngURL(data.mask);
    download('mask-link', data.mask, 'image/png');
    download('zip-link', data.archive, 'application/zip');
    const m = data.metrics;
    byId('summary').textContent = 'Средний Pearson ' + m.pearson_mean.toFixed(4) +
      ' · global SSIM ' + m.ssim_global_mean.toFixed(4) + ' · синтез ' + m.synthesis_ms.toFixed(1) + ' мс';
    byId('planes').replaceChildren(...depths.map((depth, i) => {
      const article = document.createElement('article');
      article.className = 'card';
      article.innerHTML = '<h3>Плоскость ' + (i + 1) + ' · ' + depth +
        ' мм</h3><div class="duo"><figure><figcaption>Цель</figcaption><img alt="Цель"></figure>' +
        '<figure><figcaption>Реконструкция</figcaption><img alt="Реконструкция"></figure></div><span></span>';
      const imgs = article.querySelectorAll('img');
      imgs[0].src = pngURL(data.targets[i]);
      imgs[1].src = pngURL(data.reconstructions[i]);
      article.querySelector('span').textContent = 'Pearson ' + m.pearson[i].toFixed(4) +
        ' · SSIM ' + m.ssim_global[i].toFixed(4);
      return article;
    }));
    byId('results').hidden = false;
    byId('status').textContent = 'Готово. Маска рассчитана и проверена после сохранения в PNG8.';
  } catch (e) { error(e.message); }
  finally { lock(false); }
});

