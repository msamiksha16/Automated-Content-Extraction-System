import { useState } from 'react'
import {
  ArrowRight, ArrowUpRight, Atom, BookOpen, Check, ChevronDown,
  CircleHelp, Clock3, Copy, FileText, FlaskConical, GraduationCap, Lightbulb,
  Link2, LoaderCircle, Menu, Play, RotateCcw, Sparkles, ThumbsDown, ThumbsUp,
  WandSparkles, X,
} from 'lucide-react'

const STYLES = [
  { value: 'summary_short', label: 'Quick recap', meta: 'A quick, focused overview' },
  { value: 'summary_medium', label: 'Balanced notes', meta: 'The essentials with context' },
  { value: 'summary_long', label: 'Deep dive', meta: 'Detailed notes, step by step' },
  { value: 'summary_formal', label: 'Class notes', meta: 'Clear, formal explanations' },
  { value: 'summary_simple', label: 'Keep it simple', meta: 'Friendly, everyday language' },
]

const NAV = [
  { label: 'Summarizer', icon: Sparkles },
  { label: 'My library', icon: BookOpen },
]

function IconMark() {
  return <span className="brand-mark"><span /><span /><span /><span /></span>
}

function App() {
  const [url, setUrl] = useState('')
  const [style, setStyle] = useState('summary_medium')
  const [loading, setLoading] = useState(false)
  const [regenerating, setRegenerating] = useState(false)
  const [data, setData] = useState(null)
  const [tab, setTab] = useState('summary')
  const [error, setError] = useState('')
  const [vote, setVote] = useState(null)
  const [feedback, setFeedback] = useState('')
  const [copied, setCopied] = useState(false)
  const [menuOpen, setMenuOpen] = useState(false)

  const chosenStyle = STYLES.find((item) => item.value === style) || STYLES[1]

  async function requestJson(path, body) {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    const result = await response.json()
    if (!response.ok || result.error) throw new Error(result.error || 'Something went wrong. Please try again.')
    return result
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setError('')
    setFeedback('')
    setVote(null)
    setLoading(true)
    setTab('summary')
    try {
      const result = await requestJson('/youtube', { url: url.trim(), user_id: getUserId(), summary_type: style })
      setData(result)
    } catch (exception) {
      setError(exception.message || 'Could not create your notes. Check the link and try again.')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  async function sendFeedback(value) {
    setVote(value)
    setFeedback('')
    try {
      await requestJson('/feedback', {
        user_id: getUserId(), video_url: url.trim(), summary: data.summary,
        feedback: value, variant: style,
      })
      setFeedback('Thanks — your feedback makes future summaries better.')
    } catch (exception) {
      setVote(null)
      setFeedback(exception.message || 'Could not save your feedback.')
    }
  }

  async function handleRegenerate() {
    setRegenerating(true)
    setError('')
    try {
      const result = await requestJson('/resummarize', { url: url.trim(), user_id: getUserId(), summary_type: style })
      setData((previous) => ({ ...previous, summary: result.new_summary, quality_score: result.quality_score }))
      setFeedback('A fresh version is ready. Does this one click?')
      setVote(null)
    } catch (exception) {
      setError(exception.message || 'Could not make another version.')
    } finally {
      setRegenerating(false)
    }
  }

  async function copyNotes() {
    try {
      await navigator.clipboard.writeText(data.summary)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      setError('Clipboard access is unavailable in this browser.')
    }
  }

  const tabs = [
    { id: 'summary', label: 'Summary', icon: FileText, count: null },
    { id: 'formulas', label: 'Formulas', icon: Atom, count: data?.formulas?.length },
    { id: 'examples', label: 'Examples', icon: Lightbulb, count: data?.examples?.length },
    { id: 'details', label: 'Video details', icon: Play, count: null },
  ]

  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? 'sidebar-open' : ''}`}>
        <a className="brand" href="#top" aria-label="Lesson home"><IconMark /><span>lesson<span className="brand-period">.</span></span></a>
        <div className="workspace-label">YOUR WORKSPACE <ChevronDown size={14} /></div>
        <nav className="side-nav" aria-label="Main navigation">
          {NAV.map(({ label, icon: Icon }) => (
            <button key={label} className={`nav-item ${label === 'Summarizer' ? 'nav-active' : ''}`} onClick={() => setMenuOpen(false)}>
              <Icon size={18} strokeWidth={1.8} /><span>{label}</span>{label === 'My library' && <span className="nav-soon">soon</span>}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-note"><div className="note-icon"><GraduationCap size={17} /></div><p>Big ideas, made easier.</p><span>One lesson at a time.</span><div className="note-doodle">✳</div></div>
        <div className="sidebar-profile"><div className="avatar">S</div><div><strong>Study space</strong><span>Your personal desk</span></div><CircleHelp size={18} className="help-icon" /></div>
      </aside>

      {menuOpen && <button className="mobile-scrim" aria-label="Close navigation" onClick={() => setMenuOpen(false)} />}

      <main className="main-area" id="top">
        <header className="topbar">
          <button className="mobile-menu" aria-label="Open menu" onClick={() => setMenuOpen(true)}><Menu size={21} /></button>
          <div className="breadcrumb">Your workspace <span>/</span> <strong>Summarizer</strong></div>
          <a className="topbar-link" href="#how-it-works">How it works <ArrowUpRight size={15} /></a>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div><div className="eyebrow"><span className="eyebrow-dot" /> YOUR PERSONAL STUDY SHORTCUT</div><h1>Make every lesson <span>click.</span></h1><p className="welcome-copy">Turn a math video into notes you’ll actually want to come back to.</p></div>
            <div className="welcome-doodle" aria-hidden="true"><span className="doodle-ring">∑</span><span className="doodle-star">✳</span><span className="doodle-dot" /></div>
          </section>

          <section className="input-card" aria-labelledby="input-heading">
            <div className="card-kicker"><span className="step-number">01</span><span>START WITH A VIDEO</span><span className="kicker-line" /></div>
            <div className="input-heading-row"><div><h2 id="input-heading">Drop your lesson link</h2><p>Got a YouTube math lesson? We’ll take it from here.</p></div><div className="link-illustration"><Link2 size={21} /></div></div>
            <form onSubmit={handleSubmit}>
              <label className="input-label" htmlFor="video-url">YOUTUBE VIDEO LINK</label>
              <div className="url-field"><span className="url-icon"><Link2 size={18} /></span><input id="video-url" type="url" placeholder="https://youtube.com/watch?v=..." value={url} onChange={(event) => setUrl(event.target.value)} required /><span className="url-hint">↵</span></div>
              <div className="form-bottom"><label className="style-select-wrap"><span className="select-icon"><WandSparkles size={17} /></span><select aria-label="Summary style" value={style} onChange={(event) => setStyle(event.target.value)}>{STYLES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select><span className="select-meta">{chosenStyle.meta}</span><ChevronDown size={16} className="select-chevron" /></label><button className="generate-button" type="submit" disabled={loading}>{loading ? <><LoaderCircle size={17} className="spin" /> Working on it</> : <>Make my notes <ArrowRight size={17} /></>}</button></div>
            </form>
            <div className="privacy-hint"><span className="sparkle-mini">✳</span> No setup, no fuss. Just paste and learn.</div>
          </section>

          {error && <div className="error-banner" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={18} /></button></div>}

          {loading && <section className="loading-card" aria-live="polite"><div className="loading-orbit"><LoaderCircle size={25} className="spin" /></div><div><strong>We’re getting the good bits…</strong><p>Transcribing your video and turning it into thoughtful notes.</p></div><span className="loading-time"><Clock3 size={14} /> Usually takes a minute or two</span></section>}

          {data && !loading && <section className="results-section" aria-live="polite">
            <div className="results-heading"><div><div className="eyebrow result-eyebrow"><span className="success-dot" /> YOUR NOTES ARE READY</div><h2>Here’s the big picture.</h2></div><button className="copy-button" onClick={copyNotes}>{copied ? <Check size={16} /> : <Copy size={16} />}{copied ? 'Copied' : 'Copy notes'}</button></div>
            <div className="results-card">
              <div className="tabs" role="tablist" aria-label="Summary sections">{tabs.map(({ id, label, icon: Icon, count }) => <button key={id} role="tab" aria-selected={tab === id} className={`tab-button ${tab === id ? 'tab-active' : ''}`} onClick={() => setTab(id)}><Icon size={16} /><span>{label}</span>{typeof count === 'number' && <span className="tab-count">{count}</span>}</button>)}</div>
              <div className="result-content">
                {tab === 'summary' && <><div className="summary-meta"><span className="summary-chip"><Sparkles size={13} /> {chosenStyle.label}</span>{data.video_details?.channel && <span className="video-channel">from {data.video_details.channel}</span>}</div><div className="summary-copy">{regenerating ? <div className="regenerate-state"><LoaderCircle size={19} className="spin" /> Giving your notes a fresh take…</div> : data.summary?.split(/\n+/).filter(Boolean).map((paragraph, index) => <p key={index}>{paragraph}</p>)}</div>
                  {typeof data.quality_score === 'number' && <div className="quality-row"><span className="quality-icon"><Check size={13} /></span><span>Content match</span><div className="quality-track"><span style={{ width: `${Math.max(0, Math.min(100, Math.round(data.quality_score * 100)))}%` }} /></div><strong>{Math.max(0, Math.min(100, Math.round(data.quality_score * 100)))}%</strong><span className="quality-help" title="How closely the notes reflect the original lesson.">ⓘ</span></div>}
                </>}
                {tab === 'formulas' && <ContentList items={data.formulas} empty="No formulas surfaced in this lesson. Keep watching for the big ideas." type="formula" />}
                {tab === 'examples' && <ContentList items={data.examples} empty="No worked examples were picked up this time." type="example" />}
                {tab === 'details' && <VideoDetails details={data.video_details} url={url} />}
                {tab === 'summary' && <div className="feedback-area"><div className="feedback-copy"><strong>Did these notes help?</strong><span>A little feedback goes a long way.</span></div><div className="feedback-actions"><button aria-label="These notes helped" className={vote === 1 ? 'voted' : ''} onClick={() => sendFeedback(1)} disabled={vote !== null}><ThumbsUp size={16} /></button><button aria-label="These notes need work" className={vote === 0 ? 'voted' : ''} onClick={() => sendFeedback(0)} disabled={vote !== null}><ThumbsDown size={16} /></button><span>{feedback}</span>{vote === 0 && <button className="regenerate-button" onClick={handleRegenerate} disabled={regenerating}>{regenerating ? <LoaderCircle size={14} className="spin" /> : <RotateCcw size={14} />} Try another version</button>}</div></div>}
              </div>
            </div>
          </section>}

          {!data && !loading && <section className="empty-state"><div className="empty-art"><div className="art-sheet sheet-back"><span className="art-line" /><span className="art-line short" /><span className="art-equation">x² + y²</span></div><div className="art-sheet sheet-front"><div className="art-badge"><Sparkles size={14} /></div><span className="art-line" /><span className="art-line short" /><span className="art-line medium" /><span className="art-equation purple-equation">∫ f(x) dx</span><span className="art-line" /></div><span className="art-orbit" /><span className="art-star">✳</span></div><h3>Your next “oh, I get it” moment starts here.</h3><p>Paste a video above and we’ll turn the hard bits into something that sticks.</p><div className="empty-pills"><span><FlaskConical size={14} /> Clear explanations</span><span><Atom size={14} /> Key formulas</span><span><BookOpen size={14} /> Study-ready notes</span></div></section>}

          <section className="how-section" id="how-it-works"><div className="how-title"><div className="eyebrow">A SMALLER STEP TO A BIG IDEA</div><h2>Three steps. One clearer picture.</h2></div><div className="steps-grid"><div className="step-card"><span className="step-index">01</span><div className="step-icon icon-peach"><Link2 size={19} /></div><h3>Paste a lesson</h3><p>Any YouTube math video you’ve been meaning to get through.</p></div><div className="step-card"><span className="step-index">02</span><div className="step-icon icon-lavender"><WandSparkles size={19} /></div><h3>Pick your pace</h3><p>Quick recap, deep dive, or something in between.</p></div><div className="step-card"><span className="step-index">03</span><div className="step-icon icon-green"><BookOpen size={19} /></div><h3>Get the good bits</h3><p>Come away with clear notes, examples, and the key formulas.</p></div></div></section>
          <footer className="footer"><a className="footer-brand" href="#top"><IconMark /> lesson<span className="brand-period">.</span></a><span>A little more clarity, every day. <span className="footer-spark">✳</span></span></footer>
        </div>
      </main>
    </div>
  )
}

function ContentList({ items, empty, type }) {
  if (!items?.length) return <div className="tab-empty"><span>{type === 'formula' ? <Atom size={19} /> : <Lightbulb size={19} />}</span><strong>Nothing to collect just yet.</strong><p>{empty}</p></div>
  return <div className={`content-list ${type === 'formula' ? 'formula-list' : ''}`}>{items.map((item, index) => <article className="content-item" key={`${type}-${index}`}><span className="content-index">{String(index + 1).padStart(2, '0')}</span><p>{item}</p>{type === 'formula' && <span className="formula-tag">FORMULA</span>}</article>)}</div>
}

function VideoDetails({ details = {}, url }) {
  return <div className="video-details"><div className="video-preview"><div className="play-bubble"><Play size={19} fill="currentColor" /></div><span>LESSON SOURCE</span></div><div className="video-details-content"><div className="detail-eyebrow"><span className="success-dot" /> VIDEO DETAILS</div><h3>{details.title || 'Your math lesson'}</h3><dl><div><dt>Creator</dt><dd>{details.channel || 'Not available'}</dd></div><div><dt>Length</dt><dd>{formatDuration(details.duration)}</dd></div><div><dt>Published</dt><dd>{formatDate(details.upload_date)}</dd></div></dl><a className="source-link" href={url} target="_blank" rel="noreferrer">Open original video <ArrowUpRight size={15} /></a></div></div>
}

function formatDuration(value) {
  if (value == null || value === 'Unknown Duration') return 'Not available'
  const seconds = Number(value)
  if (!Number.isFinite(seconds)) return String(value)
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainingSeconds = seconds % 60
  return hours ? `${hours}h ${minutes}m` : `${minutes}m ${remainingSeconds}s`
}

function formatDate(value) {
  if (!value || value === 'Unknown Date') return 'Not available'
  if (/^\d{8}$/.test(String(value))) return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}`
  return String(value)
}

function getUserId() {
  const key = 'lesson_summarizer_uid'
  let id = localStorage.getItem(key)
  if (!id) {
    id = `student_${Math.random().toString(36).slice(2, 13)}`
    localStorage.setItem(key, id)
  }
  return id
}

export default App
