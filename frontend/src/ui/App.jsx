import { useCallback, useEffect, useRef, useState } from 'react';

const api = async (path, options = {}) => {
  const headers = new Headers(options.headers || {});
  const method = (options.method || 'GET').toUpperCase();
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    const token = document.cookie.split('; ').find((part) => part.startsWith('XSRF-TOKEN='))?.split('=').slice(1).join('=');
    if (token) headers.set('X-XSRF-TOKEN', decodeURIComponent(token));
  }
  const response = await fetch(`/api${path}`, { ...options, headers, credentials: 'same-origin' });
  const body = response.status === 204 ? null : await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(body?.error || (response.status === 401 ? 'Please sign in to continue.' : `Request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  return body;
};

const dateLabel = (value) => value ? new Date(value).toLocaleString() : '—';
const roleLabel = (roles = []) => roles.map((role) => role.replace('ROLE_', '')).join(' · ');
const canEdit = (user) => user?.roles?.some((role) => ['ROLE_ADMIN', 'ROLE_EDITOR'].includes(role));
const isAdmin = (user) => user?.roles?.includes('ROLE_ADMIN');

function answerParagraphs(value) {
  const text = String(value || '').trim().replace(/^(?:Based on (?:the )?document|Key takeaways|Summary of the key points):\s*/i, '');
  if (!text) return [];
  const bullets = text.split(/(?:^|\n)\s*[•▪]\s*/).map((part) => part.trim()).filter(Boolean);
  if (bullets.length > 1) return [{ type: 'list', items: bullets }];
  return text.split(/\n\s*\n/).map((part) => ({ type: 'paragraph', text: part.trim() })).filter((part) => part.text);
}

function MessageView({ message, user }) {
  const paragraphs = message.role === 'assistant' ? answerParagraphs(message.text) : [{ type: 'paragraph', text: message.text }];

  return <article className={`message message-${message.role}`}>
    <div className="message-avatar">{message.role === 'user' ? user.username.slice(0, 1).toUpperCase() : 'g'}</div>
    <div className="message-content">
      <div className="message-label">{message.role === 'user' ? 'YOU' : message.grounded === false ? 'GATHER · RESPONSE' : 'GATHER · ANSWER'}</div>
      <div className="answer-body">{paragraphs.map((part, index) => part.type === 'list'
        ? <ul key={index}>{part.items.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul>
        : <p key={index}>{part.text}</p>)}</div>
    </div>
  </article>;
}

function App() {
  const [user, setUser] = useState(null);
  const [booting, setBooting] = useState(true);
  const [loginError, setLoginError] = useState('');
  const [loginHint, setLoginHint] = useState('');
  const [collections, setCollections] = useState([]);
  const [selectedId, setSelectedId] = useState('');
  const selected = collections.find((collection) => collection.id === selectedId) || null;
  const [documents, setDocuments] = useState([]);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [dialogOpen, setDialogOpen] = useState(false);
  const [collectionName, setCollectionName] = useState('');
  const [audit, setAudit] = useState([]);
  const [view, setView] = useState('library');
  const fileInput = useRef(null);
  const messagesEnd = useRef(null);
  const selectedIdRef = useRef(selectedId);
  selectedIdRef.current = selectedId;

  const refreshCollections = useCallback(async (keepSelection = true) => {
    const next = await api('/collections');
    setCollections(next);
    setSelectedId((current) => keepSelection && next.some((item) => item.id === current) ? current : (next[0]?.id || ''));
  }, []);

  const refreshDocuments = useCallback(async (id = selectedId) => {
    if (!id) { setDocuments([]); return; }
    const next = await api(`/collections/${id}/documents`);
    if (selectedIdRef.current === id) setDocuments(next);
  }, [selectedId]);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const health = await fetch('/api/health', { credentials: 'same-origin' })
          .then((response) => response.json())
          .catch(() => null);
        if (alive && health?.authOneTimeAdminCredentials === 'true') {
          const userName = health.authAdminUsername || 'admin';
          setLoginHint(`Temporary local credentials are active for ${userName}. Check backend startup logs for the generated password or set RAG_ADMIN_PASSWORD.`);
        }
        await api('/csrf');
        const session = await api('/me');
        if (!alive) return;
        setUser(session);
        await refreshCollections(false);
      } catch (reason) {
        if (reason.status !== 401 && alive) setError(reason.message);
      } finally { if (alive) setBooting(false); }
    })();
    return () => { alive = false; };
  }, [refreshCollections]);

  useEffect(() => { refreshDocuments().catch((reason) => setError(reason.message)); }, [selectedId, refreshDocuments]);

  useEffect(() => {
    if (!selectedId || !documents.some((document) => document.status === 'PROCESSING')) return undefined;
    const timer = window.setInterval(() => refreshDocuments().catch((reason) => setError(reason.message)), 2500);
    return () => window.clearInterval(timer);
  }, [selectedId, documents, refreshDocuments]);

  useEffect(() => { messagesEnd.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }); }, [messages, busy]);

  const signIn = async (event) => {
    event.preventDefault(); setLoginError(''); setBusy(true);
    const form = event.currentTarget;
    const username = form.username.value;
    const password = form.password.value;
    try {
      const token = await api('/csrf');
      const body = new URLSearchParams({ username, password, _csrf: token.token });
      const response = await fetch('/login', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-XSRF-TOKEN': token.token }, body });
      if (!response.ok) throw new Error(response.status === 401 ? 'That username and password were not accepted.' : 'Sign in could not be completed.');
      await api('/csrf');
      const session = await api('/me'); setUser(session); await refreshCollections(false);
    } catch (reason) { setLoginError(reason.message); }
    finally { setBusy(false); }
  };

  const signOut = async () => {
    try { await api('/logout', { method: 'POST' }); } catch { /* Session may already have expired. */ }
    setUser(null); setCollections([]); setSelectedId(''); setDocuments([]); setMessages([]); setView('library');
  };

  const createCollection = async (event) => {
    event.preventDefault(); setWorking(true); setError('');
    try {
      const created = await api('/collections', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: collectionName.trim() }) });
      setDialogOpen(false); setCollectionName(''); await refreshCollections(false); setSelectedId(created.id); setMessages([]);
    } catch (reason) { setError(reason.message); }
    finally { setWorking(false); }
  };

  const deleteCollection = async () => {
    if (!selected || !window.confirm(`Delete “${selected.name}” and all of its documents?`)) return;
    setWorking(true); setError('');
    try { await api(`/collections/${selected.id}`, { method: 'DELETE' }); setMessages([]); await refreshCollections(false); }
    catch (reason) { setError(reason.message); }
    finally { setWorking(false); }
  };

  const uploadFiles = async (event) => {
    const files = [...event.target.files]; event.target.value = '';
    if (!selected || !files.length) return;
    const collectionId = selected.id;
    setWorking(true); setError('');
    try {
      for (const file of files) {
        const form = new FormData(); form.append('file', file);
        try { await api(`/collections/${collectionId}/documents`, { method: 'POST', body: form }); }
        catch (reason) { setError(`${file.name}: ${reason.message}`); }
      }
      await refreshDocuments(collectionId).catch((reason) => setError(reason.message));
    } finally {
      setWorking(false);
    }
  };

  const deleteDocument = async (document) => {
    if (!window.confirm(`Remove “${document.name}” and its indexed content?`)) return;
    try { await api(`/collections/${selected.id}/documents/${document.id}`, { method: 'DELETE' }); await refreshDocuments(selected.id); }
    catch (reason) { setError(reason.message); }
  };

  const ask = async (event) => {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || !selected || busy) return;
    const collectionId = selected.id;
    setQuestion(''); setMessages((current) => [...current, { role: 'user', text: prompt }]); setBusy(true); setError('');
    try {
      const answer = await api(`/collections/${collectionId}/chat`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: prompt }) });
      if (selectedIdRef.current === collectionId) {
        setMessages((current) => [...current, { role: 'assistant', text: answer.answer, citations: answer.citations || answer.sources || [], grounded: answer.grounded }]);
      }
    } catch (reason) { if (selectedIdRef.current === collectionId) setError(reason.message); }
    finally { setBusy(false); }
  };

  const openAudit = async () => {
    setView('audit'); setError('');
    try { const page = await api('/admin/audit?limit=100'); setAudit(page.items); }
    catch (reason) { setError(reason.message); }
  };

  if (booting) return <div className="boot"><span className="brand-mark">g</span><span>Preparing your workspace…</span></div>;
  if (!user) return <main className="login-screen">
    <section className="login-panel"><div className="brand-lockup"><span className="brand-mark">g</span><span>gather<span className="brand-period">.</span></span></div>
      <div className="eyebrow">YOUR DOCUMENT INTELLIGENCE WORKSPACE</div><h1>Good knowledge starts with good questions.</h1><p className="muted">Sign in to search trusted documents and get answers with their sources.</p>
      {loginHint && <p className="muted">{loginHint}</p>}
      <form className="login-form" onSubmit={signIn}><label>Username<input name="username" autoComplete="username" required autoFocus /></label><label>Password<input name="password" type="password" autoComplete="current-password" required /></label>{loginError && <div className="inline-error">{loginError}</div>}<button className="button primary full" disabled={busy}>{busy ? 'Signing in…' : 'Sign in'} <span>→</span></button></form>
      <div className="login-foot"><span className="secure-dot" /> Secure session · CSRF protected</div>
    </section><aside className="login-art"><div className="art-orbit orbit-a"/><div className="art-orbit orbit-b"/><div className="art-card"><span className="eyebrow">GROUNDED ANSWERS</span><strong>Every answer<br />has a source.</strong><p>Keep your company knowledge connected, organized, and easy to ask.</p><div className="art-pills"><span>PDF</span><span>DOCX</span><span>TXT</span></div></div><div className="art-foot">A clearer way to work with what you know.</div></aside>
  </main>;

  return <div className="app-shell">
    <aside className="sidebar"><a className="brand-lockup" href="#library" onClick={() => setView('library')}><span className="brand-mark">g</span><span>gather<span className="brand-period">.</span></span></a>
      <div className="workspace-switch"><span className="workspace-avatar">{user.username.slice(0, 1).toUpperCase()}</span><span><b>{user.username}</b><small>{roleLabel(user.roles)}</small></span><span className="chevron">⌄</span></div>
      <div className="side-label">WORKSPACE</div><button className={`nav-link ${view === 'library' ? 'active' : ''}`} onClick={() => setView('library')}><span>▤</span>Knowledge library</button>
      {isAdmin(user) && <button className={`nav-link ${view === 'audit' ? 'active' : ''}`} onClick={openAudit}><span>◷</span>Audit activity</button>}
      <div className="collection-heading"><span className="side-label">COLLECTIONS</span>{canEdit(user) && <button className="small-icon" onClick={() => { setError(''); setDialogOpen(true); }} aria-label="Create collection">＋</button>}</div>
      <div className="side-collections">{collections.map((collection) => <button key={collection.id} className={`side-collection ${selectedId === collection.id && view === 'library' ? 'selected' : ''}`} onClick={() => { setView('library'); setSelectedId(collection.id); setMessages([]); }}><span className="collection-icon">▱</span><span className="truncate">{collection.name}</span><span className="count">{collection.documentCount}</span></button>)}</div>
      {!collections.length && <div className="sidebar-empty">No collections yet{canEdit(user) && <button onClick={() => setDialogOpen(true)}>Create your first →</button>}</div>}
      <div className="sidebar-bottom"><div className="storage-note"><span className="secure-dot"/><span><b>Your data, organized</b><small>Documents stay in their collection.</small></span></div><button className="signout" onClick={signOut}><span>↪</span> Sign out</button></div>
    </aside>

    <main className="main-area"><header className="topbar"><div className="breadcrumbs"><span>Workspace</span><span className="crumb-divider">/</span><b>{view === 'audit' ? 'Audit activity' : selected?.name || 'Knowledge library'}</b></div><div className="top-actions"><span className="api-status"><i/> Connected</span><a href="/swagger-ui.html" target="_blank" rel="noreferrer" className="icon-link" title="API documentation">API docs ↗</a></div></header>
      {error && <div className="toast-error" role="alert"><span>{error}</span><button onClick={() => setError('')} aria-label="Dismiss">×</button></div>}
      {view === 'audit' ? <section className="page-content"><div className="page-title-row"><div><div className="eyebrow">ADMINISTRATION</div><h1>Audit activity</h1><p className="muted">Security and workspace events, newest first.</p></div><button className="button outline" onClick={openAudit}>↻ Refresh</button></div><div className="audit-card"><div className="audit-head"><span>EVENT</span><span>ACTOR</span><span>RESOURCE</span><span>OUTCOME</span><span>TIME</span></div>{audit.map((event) => <div className="audit-row" key={event.eventId}><span><b>{event.action.replaceAll('_', ' ')}</b><small>#{event.eventId}</small></span><span>{event.actor}</span><span>{event.resourceType}{event.resourceId ? ` · ${event.resourceId.slice(0, 8)}` : ''}</span><span><i className={`outcome outcome-${event.outcome.toLowerCase()}`} />{event.outcome}</span><span>{dateLabel(event.occurredAt)}</span></div>)}{!audit.length && <div className="blank-state compact">No audit events found.</div>}</div></section> : <>
        <section className="welcome-banner"><div><div className="eyebrow">GATHER YOUR KNOWLEDGE</div><h1>Answers start here<span>.</span></h1><p>Bring your documents together. Ask better questions. Get answers you can trust.</p></div><div className="welcome-glyph">g</div><div className="banner-orb"/></section>
        {!selected ? <section className="blank-state"><div className="blank-icon">▱</div><h2>Your knowledge starts with a collection</h2><p>Create a collection to keep documents grouped around a team, project, or topic.</p>{canEdit(user) && <button className="button primary" onClick={() => setDialogOpen(true)}>＋ Create collection</button>}</section> : <section className="workspace-grid">
          <div className="documents-panel"><div className="panel-heading"><div><div className="eyebrow">COLLECTION</div><h2>{selected.name}</h2></div>{isAdmin(user) && <button className="danger-icon" title="Delete collection" onClick={deleteCollection} disabled={working}>⌫</button>}</div>
            <div className="document-summary"><span className="summary-icon">▤</span><span><b>{documents.length} {documents.length === 1 ? 'document' : 'documents'}</b><small>PDF, DOCX or TXT · Up to 10 MB</small></span></div>
            <div className="document-list">{documents.map((document) => <article className="document-item" key={document.id}><span className={`file-icon file-${document.name.split('.').pop()?.toLowerCase()}`}>{document.name.split('.').pop()?.toUpperCase()}</span><span className="document-info"><b title={document.name}>{document.name}</b><small>{document.status === 'READY' ? `${document.chunkCount} passages · Added ${dateLabel(document.uploadedAt)}` : document.error || `Added ${dateLabel(document.uploadedAt)}`}</small></span><span className={`status-badge status-${document.status.toLowerCase()}`}><i/>{document.status === 'PROCESSING' ? 'Indexing' : document.status === 'READY' ? 'Ready' : 'Failed'}</span>{canEdit(user) && <button className="remove-button" onClick={() => deleteDocument(document)} aria-label={`Remove ${document.name}`}>×</button>}</article>)}
              {!documents.length && <div className="no-documents"><span>▱</span><b>No documents yet</b><small>Add your first source to begin asking questions.</small></div>}</div>
            {canEdit(user) && <><input ref={fileInput} type="file" accept=".pdf,.docx,.txt" multiple hidden onChange={uploadFiles}/><button className="button upload-button" disabled={working} onClick={() => fileInput.current?.click()}><span>↑</span>{working ? 'Working…' : 'Add documents'}</button><div className="upload-hint">Files are securely processed and indexed in the background.</div></>}
          </div>
          <div className="chat-panel"><div className="chat-header"><div><div className="eyebrow">GROUNDED CONVERSATION</div><h2>Ask your collection</h2></div><span className="source-label">⌁ {documents.filter((doc) => doc.status === 'READY').length} sources</span></div>
            <div className="chat-body">
              {!messages.length ? <div className="chat-empty"><span className="chat-spark">✳</span><h3>What would you like to know?</h3><p>Ask a question or request a summary of the documents in <b>{selected.name}</b>.</p><div className="suggestion-list">{['Summarize the key points', 'What are the main policies?', 'Find important dates and deadlines'].map((suggestion) => <button key={suggestion} onClick={() => setQuestion(suggestion)} disabled={!documents.some((doc) => doc.status === 'READY')}>“{suggestion}” <span>↗</span></button>)}</div></div>
                : <div className="message-list">{messages.map((message, index) => <MessageView key={`${index}-${message.role}`} message={message} user={user} />)}{busy && <div className="thinking"><span/><span/><span/> Gathering evidence…</div>}<div ref={messagesEnd}/></div>}
            </div>
            <form className="composer" onSubmit={ask}><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.currentTarget.form.requestSubmit(); } }} placeholder="Ask a question about this collection…" maxLength={2000} disabled={busy}/><div className="composer-bottom"><span>Answers are grounded in your collection · Enter to send</span><button className="send-button" aria-label="Send question" disabled={!question.trim() || busy}>↑</button></div></form>
          </div>
        </section>}
      </>}
      <footer className="main-footer"><span>GATHER <i>·</i> DOCUMENT WORKSPACE</span><span>Grounded answers from your collection</span></footer>
    </main>

    {dialogOpen && <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !working) setDialogOpen(false); }}><form className="create-modal" onSubmit={createCollection}><button type="button" className="modal-close" onClick={() => setDialogOpen(false)} disabled={working} aria-label="Close">×</button><div className="modal-mark">＋</div><div className="eyebrow">NEW WORKSPACE</div><h2>Create a collection</h2><p>Keep document sets separate for focused, reliable answers.</p><label>Collection name<input value={collectionName} onChange={(event) => setCollectionName(event.target.value)} maxLength={80} placeholder="e.g. Team handbook" required autoFocus /></label>{error && <div className="inline-error">{error}</div>}<div className="modal-actions"><button type="button" className="button outline" onClick={() => { setDialogOpen(false); setError(''); }} disabled={working}>Cancel</button><button className="button primary" disabled={working || !collectionName.trim()}>{working ? 'Creating…' : 'Create collection'}</button></div></form></div>}
  </div>;
}

export default App;
