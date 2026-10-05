const state = { collections: [], documents: [], selected: null, user: null, busy: false };
const $ = (selector) => document.querySelector(selector);

function cookie(name) {
  const prefix = `${name}=`;
  const entry = document.cookie.split('; ').find((part) => part.startsWith(prefix));
  return entry ? decodeURIComponent(entry.slice(prefix.length)) : '';
}

async function getCsrf() {
  const response = await fetch('/api/csrf', { credentials: 'same-origin' });
  if (!response.ok) throw new Error('Could not initialize secure requests');
  return response.json();
}

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  const method = (options.method || 'GET').toUpperCase();
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    const token = cookie('XSRF-TOKEN');
    if (token) headers.set('X-XSRF-TOKEN', token);
  }
  const response = await fetch(`/api${path}`, { ...options, headers, credentials: 'same-origin' });
  let body = {};
  try { body = await response.json(); } catch { /* Non-JSON error response. */ }
  if (response.status === 401) showLogin();
  if (!response.ok) throw new Error(body.error || (response.status === 401 ? 'Sign in to continue' : `Request failed (${response.status})`));
  return body;
}

const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const isAdmin = () => state.user?.roles?.includes('ROLE_ADMIN');
const canEdit = () => isAdmin() || state.user?.roles?.includes('ROLE_EDITOR');

function toast(message) {
  const element = $('#toast');
  element.textContent = message;
  element.classList.add('show');
  setTimeout(() => element.classList.remove('show'), 2800);
}

function showLogin() {
  const dialog = $('#auth-dialog');
  if (!dialog.open) dialog.showModal();
  setTimeout(() => $('#auth-username').focus(), 20);
}

function renderCollections() {
  const list = $('#collection-list');
  list.innerHTML = state.collections.map((collection) => `
    <button class="collection-item ${state.selected?.id === collection.id ? 'selected' : ''}" data-id="${esc(collection.id)}">
      <span class="collection-name">Collection &nbsp; ${esc(collection.name)}</span>
      <span class="collection-count">${collection.documentCount}</span>
    </button>`).join('');
  list.querySelectorAll('button').forEach((button) => { button.onclick = () => selectCollection(button.dataset.id); });

  $('#empty-state').hidden = Boolean(state.selected);
  $('#new-collection').hidden = !canEdit();
  $('#empty-create').hidden = !canEdit();
  $('#upload-button').disabled = !state.selected || !canEdit();
  $('#upload-button').hidden = !canEdit();
  $('#question').disabled = !state.selected;
  $('#ask-button').disabled = !state.selected;
  $('#collection-title').textContent = state.selected?.name || 'Create a collection';
  $('#collection-label').textContent = state.selected ? 'ACTIVE COLLECTION' : 'GET STARTED';

  $('#document-strip').innerHTML = state.documents.map((document) => `
    <button class="doc-chip status-${esc(document.status.toLowerCase())}" data-id="${esc(document.id)}" title="${esc(document.error || document.name)}">
      ${esc(document.name)} <small>${esc(document.status.toLowerCase())}</small>${canEdit() ? '<span class="remove-doc"> x</span>' : ''}
    </button>`).join('');
  if (canEdit()) $('#document-strip').querySelectorAll('.doc-chip').forEach((button) => {
    button.onclick = () => removeDocument(button.dataset.id, button.title);
  });

  const sessionButton = $('#session-control');
  sessionButton.hidden = !state.user;
  sessionButton.textContent = state.user ? `${state.user.username} · Sign out` : '';
}

async function loadUser() {
  state.user = await api('/me');
  renderCollections();
}

async function refresh() {
  state.collections = await api('/collections');
  if (state.selected) {
    state.selected = state.collections.find((item) => item.id === state.selected.id) || null;
    state.documents = state.selected ? await api(`/collections/${state.selected.id}/documents`) : [];
  }
  renderCollections();
}

async function selectCollection(id) {
  state.selected = state.collections.find((item) => item.id === id) || null;
  state.documents = state.selected ? await api(`/collections/${id}/documents`) : [];
  $('#messages').replaceChildren();
  renderCollections();
}

async function removeDocument(id, name) {
  if (!confirm(`Remove ${name} and its indexed text from this collection?`)) return;
  try {
    await api(`/collections/${state.selected.id}/documents/${id}`, { method: 'DELETE' });
    state.documents = await api(`/collections/${state.selected.id}/documents`);
    await refresh();
  } catch (error) { toast(error.message); }
}

function openCollectionDialog() {
  $('#dialog-error').textContent = '';
  $('#collection-name').value = '';
  $('#collection-dialog').showModal();
  setTimeout(() => $('#collection-name').focus(), 20);
}

$('#new-collection').onclick = openCollectionDialog;
$('#empty-create').onclick = openCollectionDialog;
$('#cancel-collection').onclick = () => $('#collection-dialog').close();
$('#collection-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#create-button');
  button.disabled = true;
  try {
    const item = await api('/collections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: $('#collection-name').value }) });
    $('#collection-dialog').close();
    await refresh();
    await selectCollection(item.id);
  } catch (error) { $('#dialog-error').textContent = error.message; }
  finally { button.disabled = false; }
});

$('#auth-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const button = $('#auth-submit');
  button.disabled = true;
  $('#auth-error').textContent = '';
  try {
    const token = cookie('XSRF-TOKEN');
    const body = new URLSearchParams({ username: $('#auth-username').value, password: $('#auth-password').value, _csrf: token });
    const response = await fetch('/login', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-XSRF-TOKEN': token }, body });
    if (!response.ok) throw new Error(response.status === 401 ? 'Username or password is incorrect' : 'Sign in could not be completed');
    $('#auth-dialog').close();
    $('#auth-form').reset();
    await getCsrf();
    await loadUser();
    await refresh();
  } catch (error) { $('#auth-error').textContent = error.message; }
  finally { button.disabled = false; }
});

$('#session-control').onclick = async () => {
  try {
    const token = cookie('XSRF-TOKEN');
    await fetch('/logout', { method: 'POST', credentials: 'same-origin', headers: { 'X-XSRF-TOKEN': token } });
    state.user = null; state.selected = null; state.documents = []; state.collections = [];
    renderCollections(); showLogin();
  } catch (error) { toast(error.message); }
};

$('#upload-button').onclick = () => $('#file-input').click();
$('#file-input').onchange = async (event) => {
  const files = [...event.target.files];
  if (!state.selected || !files.length) return;
  const collectionId = state.selected.id;
  for (let index = 0; index < files.length; index += 3) {
    const batch = files.slice(index, index + 3);
    await Promise.allSettled(batch.map(async (file) => {
      const form = new FormData(); form.append('file', file);
      try { await api(`/collections/${collectionId}/documents`, { method: 'POST', body: form }); }
      catch (error) { toast(`${file.name}: ${error.message}`); }
    }));
  }
  event.target.value = '';
  for (let attempt = 0; attempt < 120; attempt++) {
    if (!state.selected || state.selected.id !== collectionId) break;
    state.documents = await api(`/collections/${collectionId}/documents`);
    renderCollections();
    if (state.documents.every((document) => document.status !== 'PROCESSING')) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  await refresh();
};

function addMessage(role, content, sources = []) {
  const box = document.createElement('div'); box.className = `message ${role}`;
  if (role === 'assistant') {
    const label = document.createElement('span'); label.className = 'message-label'; label.textContent = 'GATHER · GROUNDED RESPONSE'; box.append(label);
  }
  const text = document.createElement('div'); text.textContent = content; box.append(text);
  if (sources.length) {
    const citations = document.createElement('div'); citations.className = 'citations';
    for (const source of sources) {
      const card = document.createElement('div'); card.className = 'citation';
      card.innerHTML = `<strong>${esc(source.documentName)} · ${source.page ? `page ${source.page} · ` : ''}chunk ${source.chunk}</strong><p>${esc(source.excerpt)}</p>`;
      citations.append(card);
    }
    box.append(citations);
  }
  $('#messages').append(box); box.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

$('#ask-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#question'); const question = input.value.trim();
  if (!question || !state.selected || state.busy) return;
  addMessage('user', question); input.value = ''; state.busy = true; $('#ask-button').disabled = true;
  try {
    const response = await api(`/collections/${state.selected.id}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question }) });
    addMessage('assistant', response.answer, response.sources);
  } catch (error) { addMessage('assistant', error.message); }
  finally { state.busy = false; $('#ask-button').disabled = !state.selected; input.focus(); }
});

$('#question').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); $('#ask-form').requestSubmit(); }
});

(async () => {
  try { await getCsrf(); await loadUser(); await refresh(); }
  catch (error) { if (!$('#auth-dialog').open) showLogin(); }
})();
