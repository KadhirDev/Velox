import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useAuth } from './AuthContext'
import { register as apiRegister } from '../api/client'
import { Zap, Loader, Eye, EyeOff, ArrowRight, UserPlus, LogIn } from 'lucide-react'

function InputField({ label, type = 'text', value, onChange, onKeyDown, placeholder, autoFocus, autoComplete, rightSlot }) {
  const [focused, setFocused] = useState(false)
  return (
    <div style={{ marginBottom: 16 }}>
      <label style={{ display: 'block', fontSize: 12, fontWeight: 600, color: 'var(--muted)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</label>
      <div style={{ position: 'relative' }}>
        <input
          type={type}
          value={value}
          onChange={onChange}
          onKeyDown={onKeyDown}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={placeholder}
          autoFocus={autoFocus}
          autoComplete={autoComplete}
          style={{
            width: '100%', padding: '12px 14px', background: 'var(--surface2)', border: '1px solid var(--border)',
            borderRadius: 10, color: 'var(--text)', fontSize: 14, outline: 'none',
            paddingRight: rightSlot ? 42 : 14, transition: 'border-color 0.2s, box-shadow 0.2s',
            boxSizing: 'border-box',
          }}
          onMouseEnter={(e) => !focused && (e.target.style.borderColor = 'var(--accent)')}
          onMouseLeave={(e) => !focused && (e.target.style.borderColor = 'var(--border)')}
        />
        <motion.div
          initial={false}
          animate={{ scaleX: focused ? 1 : 0, opacity: focused ? 1 : 0 }}
          transition={{ duration: 0.2 }}
          style={{
            position: 'absolute', bottom: 0, left: 0, right: 0, height: 2,
            borderRadius: '0 0 10px 10px',
            background: 'linear-gradient(90deg, var(--accent), var(--accent2))',
            transformOrigin: 'left',
          }}
        />
        {rightSlot && (
          <div style={{ position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)', zIndex: 1 }}>
            {rightSlot}
          </div>
        )}
      </div>
    </div>
  )
}

export default function LoginPage() {
  const { login } = useAuth()
  const [mode, setMode] = useState('signin')
  const [form, setForm] = useState({ username: '', password: '', confirm: '' })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirm, setShowConfirm] = useState(false)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [loading, setLoading] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const updateField = (key, value) => {
    setForm((prev) => ({ ...prev, [key]: value }))
    setError(null)
    setSuccess(null)
  }

  const switchMode = (nextMode) => {
    if (loading) return
    setMode(nextMode)
    setForm({ username: '', password: '', confirm: '' })
    setError(null)
    setSuccess(null)
    setShowPassword(false)
    setShowConfirm(false)
  }

  const handleSignIn = async () => {
    const username = form.username.trim()
    const password = form.password
    if (!username || !password) { setError('Please enter your username and password.'); return }
    setLoading(true); setError(null); setSuccess(null)
    try { await login(username, password) }
    catch (e) { setError(e?.response?.data?.detail || 'Invalid username or password.') }
    finally { setLoading(false) }
  }

  const handleSignUp = async () => {
    const username = form.username.trim()
    const password = form.password
    const confirm = form.confirm
    if (!username) { setError('Username is required.'); return }
    if (username.length < 3) { setError('Username must be at least 3 characters.'); return }
    if (password.length < 6) { setError('Password must be at least 6 characters.'); return }
    if (password !== confirm) { setError('Passwords do not match.'); return }
    setLoading(true); setError(null); setSuccess(null)
    try {
      const result = await apiRegister(username, password, confirm)
      setMode('signin')
      setForm({ username: result?.username || username, password: '', confirm: '' })
      setShowPassword(false); setShowConfirm(false); setError(null)
      setSuccess(result?.message || `Account created. Sign in as "${result?.username || username}".`)
    }
    catch (e) { setError(e?.response?.data?.detail || 'Registration failed. Please try again.') }
    finally { setLoading(false) }
  }

  const handleSubmit = async () => {
    if (mode === 'signin') await handleSignIn()
    else await handleSignUp()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') { e.preventDefault(); handleSubmit() }
  }

  const handleGoogleClick = () => {
    console.log('Google OAuth coming soon')
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex', background: 'var(--bg)',
      position: 'relative', overflow: 'hidden',
    }}>
      {!isMobile && (
        <motion.aside
          initial={{ opacity: 0, x: -40 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          onMouseMove={(e) => {
            if (typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
            const r = e.currentTarget.getBoundingClientRect()
            const px = (e.clientX - r.left) / r.width - 0.5
            const py = (e.clientY - r.top) / r.height - 0.5
            e.currentTarget.style.setProperty('--px', px.toFixed(3))
            e.currentTarget.style.setProperty('--py', py.toFixed(3))
          }}
          style={{
            width: '55%', height: '100vh', position: 'fixed', left: 0, top: 0, zIndex: 1,
            background: 'linear-gradient(135deg, #070b16 0%, #0a1226 45%, #0b1030 75%, #070b16 100%)',
            display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
            padding: '48px 64px', overflow: 'hidden',
          }}
        >
          {/* ── Space/cloud atmosphere layers (pure CSS, no canvas) ── */}
          <div aria-hidden="true" className="vlx-storm-bg" />
          <div aria-hidden="true" className="vlx-storm-glow" />
          <div aria-hidden="true" className="vlx-haze" />
          <div aria-hidden="true" className="vlx-stars">
            <span style={{ left: '8%', top: '12%' }} />
            <span style={{ left: '18%', top: '68%' }} />
            <span style={{ left: '12%', top: '42%', animationDelay: '1.2s' }} />
            <span style={{ left: '28%', top: '22%', animationDelay: '2.1s' }} />
            <span style={{ left: '72%', top: '14%', animationDelay: '0.6s' }} />
            <span style={{ left: '84%', top: '58%', animationDelay: '1.8s' }} />
            <span style={{ left: '90%', top: '30%' }} />
            <span style={{ left: '64%', top: '78%', animationDelay: '2.6s' }} />
            <span style={{ left: '42%', top: '8%', animationDelay: '0.9s' }} />
            <span style={{ left: '55%', top: '88%' }} />
            <span style={{ left: '36%', top: '82%', animationDelay: '1.5s' }} />
            <span style={{ left: '78%', top: '84%', animationDelay: '2.9s' }} />
          </div>

          <motion.div
            style={{ position: 'relative', zIndex: 2, maxWidth: 560, width: '100%' }}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          >
            {/* Brand — upper-left */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
              <div
                style={{
                  width: 44, height: 44, borderRadius: 12,
                  background: 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  boxShadow: '0 0 28px rgba(99,102,241,0.45)',
                }}
              >
                <Zap size={22} color="#fff" />
              </div>
              <span style={{ fontSize: 24, fontWeight: 900, letterSpacing: '-0.03em', background: 'linear-gradient(135deg, #fff, #e0e7ff)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                Velox
              </span>
            </div>

            <p style={{ color: 'rgba(165,180,252,0.85)', fontSize: 11, fontWeight: 700, letterSpacing: '0.22em', textTransform: 'uppercase', marginBottom: 12 }}>
              Cloud Orchestration
            </p>

            <h2 style={{ fontSize: 40, lineHeight: 1.12, fontWeight: 800, letterSpacing: '-0.02em', color: '#fff', margin: '0 0 12px' }}>
              Smarter <span style={{ color: '#fff' }}>Decisions.</span>
              <br />
              Lower <span style={{ background: 'linear-gradient(135deg, #a78bfa, #6366f1)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Costs.</span>
              <br />
              <span style={{ background: 'linear-gradient(135deg, #67e8f9, #34d399)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>Greener Cloud.</span>
            </h2>

            {/* ── HERO: orbital cloud mesh ── */}
            <div className="vlx-hero" style={{ position: 'relative', width: '100%', margin: '6px 0 2px' }}>
              <svg viewBox="0 0 560 360" width="100%" height="340" role="img" aria-label="Velox intelligence core orchestrating workloads across orbiting AWS, Azure and GCP nodes" style={{ display: 'block', overflow: 'visible' }}>
                <defs>
                  <radialGradient id="vlxCoreAura" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#818cf8" stopOpacity="0.5" />
                    <stop offset="45%" stopColor="#6366f1" stopOpacity="0.22" />
                    <stop offset="75%" stopColor="#3b82f6" stopOpacity="0.08" />
                    <stop offset="100%" stopColor="#3b82f6" stopOpacity="0" />
                  </radialGradient>
                  <linearGradient id="vlxRing" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#a5b4fc" stopOpacity="0.55" />
                    <stop offset="50%" stopColor="#6366f1" stopOpacity="0.18" />
                    <stop offset="100%" stopColor="#67e8f9" stopOpacity="0.45" />
                  </linearGradient>
                  <linearGradient id="vlxLine" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#c7d2fe" stopOpacity="0.85" />
                    <stop offset="100%" stopColor="#6366f1" stopOpacity="0.25" />
                  </linearGradient>
                  <linearGradient id="vlxGcp" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#4285F4" />
                    <stop offset="38%" stopColor="#EA4335" />
                    <stop offset="68%" stopColor="#FBBC05" />
                    <stop offset="100%" stopColor="#34A853" />
                  </linearGradient>
                </defs>

                {/* faint static orbital traces */}
                <g fill="none" opacity="0.35">
                  <ellipse cx="280" cy="182" rx="238" ry="62" stroke="#4c5a8a" strokeOpacity="0.35" strokeWidth="1" />
                  <ellipse cx="280" cy="182" rx="120" ry="128" stroke="#4c5a8a" strokeOpacity="0.22" strokeWidth="1" />
                </g>

                {/* slow-rotating orbital rings */}
                <g fill="none" className="vlx-ring-rot-a">
                  <ellipse cx="280" cy="182" rx="205" ry="80" transform="rotate(-10 280 182)" stroke="url(#vlxRing)" strokeWidth="1.2" opacity="0.6" />
                </g>
                <g fill="none" className="vlx-ring-rot-b">
                  <ellipse cx="280" cy="182" rx="160" ry="110" transform="rotate(14 280 182)" stroke="url(#vlxRing)" strokeWidth="1" opacity="0.45" />
                </g>

                {/* data connections: core -> providers */}
                <g fill="none" strokeWidth="1.5">
                  <path d="M274 128 Q264 100 276 82" stroke="url(#vlxLine)" opacity="0.6" className="vlx-dash" />
                  <path d="M242 208 Q192 238 154 266" stroke="url(#vlxLine)" opacity="0.6" className="vlx-dash" />
                  <path d="M318 208 Q368 238 406 266" stroke="url(#vlxLine)" opacity="0.6" className="vlx-dash" />
                </g>

                {/* travelling particles: core -> cloud, plus feedback returns */}
                <g className="vlx-particle">
                  <circle r="3" fill="#c7d2fe">
                    <animateMotion dur="3s" repeatCount="indefinite" path="M274 128 Q264 100 276 82" />
                  </circle>
                  <circle r="2.4" fill="#818cf8">
                    <animateMotion dur="4.2s" begin="1.1s" repeatCount="indefinite" calcMode="linear" keyPoints="1;0" keyTimes="0;1" path="M274 128 Q264 100 276 82" />
                  </circle>
                  <circle r="3" fill="#a5b4fc">
                    <animateMotion dur="3.6s" begin="0.5s" repeatCount="indefinite" path="M242 208 Q192 238 154 266" />
                  </circle>
                  <circle r="2.4" fill="#818cf8">
                    <animateMotion dur="5s" begin="2s" repeatCount="indefinite" calcMode="linear" keyPoints="1;0" keyTimes="0;1" path="M242 208 Q192 238 154 266" />
                  </circle>
                  <circle r="3" fill="#67e8f9">
                    <animateMotion dur="3.3s" begin="1.4s" repeatCount="indefinite" path="M318 208 Q368 238 406 266" />
                  </circle>
                  <circle r="2.4" fill="#818cf8">
                    <animateMotion dur="4.6s" begin="0.3s" repeatCount="indefinite" calcMode="linear" keyPoints="1;0" keyTimes="0;1" path="M318 208 Q368 238 406 266" />
                  </circle>
                </g>

                {/* central Velox intelligence core (parallax layer 1) */}
                <g className="vlx-px-core">
                  <g className="vlx-corefloat">
                    <circle cx="280" cy="180" r="88" fill="url(#vlxCoreAura)" className="vlx-corebreath" />
                    <circle cx="280" cy="180" r="62" fill="rgba(32,44,82,0.55)" stroke="rgba(148,163,255,0.35)" strokeWidth="1.2" />
                    <circle cx="280" cy="180" r="62" fill="none" stroke="rgba(165,180,252,0.2)" strokeWidth="5" className="vlx-pulse" />
                    <circle cx="280" cy="180" r="44" fill="#232f5c" stroke="rgba(199,210,254,0.5)" strokeWidth="1.2" />
                    <ellipse cx="264" cy="164" rx="18" ry="10" fill="rgba(199,210,254,0.18)" />
                    <path d="M287 158 L271 186 L282 186 L276 204 L293 174 L284 174 Z" fill="#ffffff" className="vlx-corebolt" />
                    <text x="280" y="252" textAnchor="middle" fill="#e0e7ff" fontSize="13" letterSpacing="5" fontWeight="800">VELOX</text>
                  </g>
                </g>

                {/* orbiting provider satellites (parallax layer 2) */}
                <g className="vlx-px-nodes">
                  <g className="vlx-sat-a">
                    <circle cx="280" cy="52" r="36" fill="none" stroke="#f59e0b" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="4 6" className="vlx-sat-ring" />
                    <circle cx="280" cy="52" r="22" fill="#141c33" stroke="#f59e0b" strokeOpacity="0.7" strokeWidth="1.5" />
                    <circle cx="280" cy="52" r="22" fill="none" stroke="#f59e0b" strokeOpacity="0.25" strokeWidth="5" className="vlx-pulse" />
                    <circle cx="280" cy="52" r="6.5" fill="#f59e0b" />
                    <text x="280" y="102" textAnchor="middle" fill="rgba(255,255,255,0.65)" fontSize="11" letterSpacing="2" fontWeight="700">AWS</text>
                  </g>
                  <g className="vlx-sat-b">
                    <circle cx="140" cy="280" r="36" fill="none" stroke="#3b82f6" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="4 6" className="vlx-sat-ring" />
                    <circle cx="140" cy="280" r="22" fill="#141c33" stroke="#3b82f6" strokeOpacity="0.7" strokeWidth="1.5" />
                    <circle cx="140" cy="280" r="22" fill="none" stroke="#3b82f6" strokeOpacity="0.25" strokeWidth="5" className="vlx-pulse" />
                    <circle cx="140" cy="280" r="6.5" fill="#3b82f6" />
                    <text x="140" y="330" textAnchor="middle" fill="rgba(255,255,255,0.65)" fontSize="11" letterSpacing="2" fontWeight="700">AZURE</text>
                  </g>
                  <g className="vlx-sat-c">
                    <circle cx="420" cy="280" r="36" fill="none" stroke="#34A853" strokeOpacity="0.3" strokeWidth="1" strokeDasharray="4 6" className="vlx-sat-ring" />
                    <circle cx="420" cy="280" r="22" fill="#141c33" stroke="url(#vlxGcp)" strokeOpacity="0.85" strokeWidth="1.5" />
                    <circle cx="420" cy="280" r="22" fill="none" stroke="#34A853" strokeOpacity="0.22" strokeWidth="5" className="vlx-pulse" />
                    <circle cx="413" cy="276" r="3.6" fill="#4285F4" />
                    <circle cx="420" cy="275" r="3.6" fill="#EA4335" />
                    <circle cx="426" cy="281" r="3.6" fill="#FBBC05" />
                    <circle cx="417" cy="284" r="3.6" fill="#34A853" />
                    <text x="420" y="330" textAnchor="middle" fill="rgba(255,255,255,0.65)" fontSize="11" letterSpacing="2" fontWeight="700">GCP</text>
                  </g>
                </g>

                {/* decorative micro-labels + ambient dust */}
                <g className="vlx-micro" fontSize="8.5" letterSpacing="2.5" fontWeight="600" fill="rgba(199,210,254,0.32)">
                  <text x="66" y="150">OPTIMIZE</text>
                  <text x="474" y="150">SCHEDULE</text>
                  <text x="58" y="238">ANALYZE</text>
                  <text x="492" y="238">DEPLOY</text>
                </g>
                <g fill="rgba(199,210,254,0.5)" className="vlx-dust">
                  <circle cx="200" cy="120" r="1.4" />
                  <circle cx="372" cy="110" r="1.2" />
                  <circle cx="408" cy="180" r="1.6" />
                  <circle cx="152" cy="180" r="1.6" />
                  <circle cx="238" cy="292" r="1.3" />
                  <circle cx="330" cy="296" r="1.3" />
                </g>
              </svg>
            </div>

          </motion.div>

          {/* ── Left-panel animation stylesheet (transform/opacity only) ── */}
          <style>{`
            .vlx-storm-bg { position: absolute; inset: 0; z-index: 0; pointer-events: none;
              background:
                radial-gradient(900px 480px at 50% 32%, rgba(99,102,241,0.15), transparent 65%),
                radial-gradient(700px 420px at 18% 82%, rgba(59,130,246,0.10), transparent 65%),
                radial-gradient(720px 420px at 86% 76%, rgba(168,85,247,0.09), transparent 65%),
                radial-gradient(500px 300px at 50% 55%, rgba(34,211,238,0.05), transparent 70%);
            }
            .vlx-storm-glow { position: absolute; inset: 0; z-index: 0; pointer-events: none; opacity: 0.5;
              background: radial-gradient(440px 250px at 50% 46%, rgba(129,140,248,0.17), transparent 70%);
              animation: vlxBreathe 7s ease-in-out infinite;
            }
            .vlx-haze { position: absolute; inset: 0; z-index: 0; pointer-events: none; opacity: 0.6;
              background: radial-gradient(620px 200px at 50% 62%, rgba(99,102,241,0.07), transparent 70%);
            }
            .vlx-stars { position: absolute; inset: 0; z-index: 0; pointer-events: none; }
            .vlx-stars span { position: absolute; width: 2px; height: 2px; border-radius: 50%;
              background: rgba(199,210,254,0.8); opacity: 0.25; animation: vlxTwinkle 5s ease-in-out infinite; }
            .vlx-corefloat { transform-box: fill-box; transform-origin: center;
              filter: drop-shadow(0 0 28px rgba(99,102,241,0.5));
              animation: vlxCoreFloat 10s ease-in-out infinite; }
            .vlx-corebreath { animation: vlxBreathe 6s ease-in-out infinite; }
            .vlx-corebolt { filter: drop-shadow(0 0 8px rgba(255,255,255,0.9)); }
            .vlx-ring-rot-a { transform-origin: 280px 182px; animation: vlxSpin 80s linear infinite; }
            .vlx-ring-rot-b { transform-origin: 280px 182px; animation: vlxSpinRev 120s linear infinite; }
            .vlx-sat-a { transform-box: fill-box; transform-origin: center; animation: vlxSatA 7.5s ease-in-out infinite; }
            .vlx-sat-b { transform-box: fill-box; transform-origin: center; animation: vlxSatB 9s ease-in-out infinite; }
            .vlx-sat-c { transform-box: fill-box; transform-origin: center; animation: vlxSatA 10.5s ease-in-out infinite reverse; }
            .vlx-sat-ring { animation: vlxDashSlow 14s linear infinite; }
            .vlx-pulse { transform-box: fill-box; transform-origin: center; animation: vlxPulse 3.6s ease-in-out infinite; }
            .vlx-dash { stroke-dasharray: 5 7; animation: vlxDash 2.8s linear infinite; }
            .vlx-micro { animation: vlxTwinkle 7s ease-in-out infinite; }
            .vlx-dust { animation: vlxTwinkle 6s ease-in-out infinite; }
            .vlx-px-core { transform: translate3d(calc(var(--px, 0) * 12px), calc(var(--py, 0) * 9px), 0); }
            .vlx-px-nodes { transform: translate3d(calc(var(--px, 0) * -8px), calc(var(--py, 0) * -6px), 0); }
            @keyframes vlxSpin { to { transform: rotate(360deg); } }
            @keyframes vlxSpinRev { to { transform: rotate(-360deg); } }
            @keyframes vlxCoreFloat { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
            @keyframes vlxSatA { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
            @keyframes vlxSatB { 0%,100% { transform: translateY(0); } 50% { transform: translateY(4px); } }
            @keyframes vlxBreathe { 0%,100% { opacity: 0.65; } 50% { opacity: 1; } }
            @keyframes vlxPulse { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
            @keyframes vlxTwinkle { 0%,100% { opacity: 0.12; } 50% { opacity: 0.5; } }
            @keyframes vlxDash { to { stroke-dashoffset: -24; } }
            @keyframes vlxDashSlow { to { stroke-dashoffset: -100; } }
            @media (prefers-reduced-motion: reduce) {
              .vlx-corefloat, .vlx-corebreath, .vlx-ring-rot-a, .vlx-ring-rot-b,
              .vlx-sat-a, .vlx-sat-b, .vlx-sat-c, .vlx-sat-ring, .vlx-pulse,
              .vlx-dash, .vlx-micro, .vlx-dust, .vlx-stars span, .vlx-storm-glow { animation: none !important; }
              .vlx-particle { display: none; }
              .vlx-px-core, .vlx-px-nodes { transform: none; }
            }
          `}</style>
        </motion.aside>
      )}

      <motion.main
        initial={{ opacity: 0, x: isMobile ? 0 : 40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1], delay: isMobile ? 0 : 0.1 }}
        style={{
          width: isMobile ? '100%' : '45%', minHeight: '100vh',
          marginLeft: isMobile ? 0 : '55%', position: 'relative', zIndex: 2,
          background: 'var(--surface)', borderLeft: isMobile ? 'none' : '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center',
          padding: '40px 60px', boxSizing: 'border-box',
        }}
      >
        <motion.div
          style={{ width: '100%', maxWidth: 400, position: 'relative' }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          {/* Tab switcher */}
          <div style={{
            display: 'flex', background: 'var(--surface2)',
            border: '1px solid var(--border)', borderRadius: 10,
            padding: 4, marginBottom: 24, gap: 4, position: 'relative',
          }}>
            {[{ id: 'signin', label: 'Sign In', Icon: LogIn }, { id: 'signup', label: 'Sign Up', Icon: UserPlus }].map(({ id, label, Icon }) => (
              <button
                key={id}
                type="button"
                onClick={() => switchMode(id)}
                disabled={loading}
                style={{
                  flex: 1, padding: '10px 0', borderRadius: 8,
                  fontWeight: 600, fontSize: 13,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  background: mode === id ? 'linear-gradient(135deg, var(--accent), var(--accent2))' : 'transparent',
                  color: mode === id ? '#fff' : 'var(--muted)',
                  border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
                  transition: 'all 0.22s cubic-bezier(0.16,1,0.3,1)',
                  transform: mode === id ? 'scale(1.02)' : 'scale(1)',
                }}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}
          </div>

          {/* Form card */}
          <div className="card" style={{ padding: '28px 28px 24px', boxShadow: '0 24px 64px rgba(0,0,0,0.35), 0 0 0 1px rgba(99,102,241,0.06)', borderColor: 'rgba(99,102,241,0.1)', borderRadius: 16 }}>
            <motion.div
              key={mode}
              initial={{ opacity: 0, x: mode === 'signin' ? -16 : 16 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: mode === 'signin' ? 16 : -16 }}
              transition={{ duration: 0.22, ease: 'easeOut' }}
            >
              <div style={{ marginBottom: 24 }}>
                <div style={{ fontWeight: 700, fontSize: 17, marginBottom: 6 }}>
                  {mode === 'signin' ? 'Welcome back' : 'Create account'}
                </div>
                <div style={{ color: 'var(--muted)', fontSize: 13, lineHeight: 1.5 }}>
                  {mode === 'signin' ? 'Sign in with your credentials' : 'Create a new account to access Velox'}
                </div>
              </div>

              <AnimatePresence mode="wait">
                {success && (
                  <motion.div
                    key="success"
                    initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}
                    style={{
                      background: 'rgba(16,185,129,0.1)', border: '1px solid rgba(16,185,129,0.3)',
                      borderRadius: 8, padding: '10px 14px', color: 'var(--green)',
                      fontSize: 13, marginBottom: 18, display: 'flex', alignItems: 'center', gap: 8,
                    }}
                  >
                    <span>✓</span><span>{success}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <InputField
                label="Username"
                value={form.username}
                onChange={(e) => updateField('username', e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your username"
                autoFocus
                autoComplete="username"
              />

              <InputField
                label="Password"
                type={showPassword ? 'text' : 'password'}
                value={form.password}
                onChange={(e) => updateField('password', e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Enter your password"
                autoComplete={mode === 'signin' ? 'current-password' : 'new-password'}
                rightSlot={
                  <button
                    type="button"
                    onClick={() => setShowPassword((p) => !p)}
                    style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex', cursor: 'pointer' }}
                  >
                    {showPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                  </button>
                }
              />

              {mode === 'signup' && (
                <InputField
                  label="Confirm Password"
                  type={showConfirm ? 'text' : 'password'}
                  value={form.confirm}
                  onChange={(e) => updateField('confirm', e.target.value)}
                  onKeyDown={handleKeyDown}
                  placeholder="Repeat your password"
                  autoComplete="new-password"
                  rightSlot={
                    <button
                      type="button"
                      onClick={() => setShowConfirm((p) => !p)}
                      style={{ background: 'none', color: 'var(--muted)', padding: 0, border: 'none', borderRadius: 0, display: 'flex', cursor: 'pointer' }}
                    >
                      {showConfirm ? <EyeOff size={15} /> : <Eye size={15} />}
                    </button>
                  }
                />
              )}

              <AnimatePresence>
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
                    style={{
                      background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
                      borderRadius: 8, padding: '10px 14px', color: '#fca5a5',
                      fontSize: 12, marginBottom: 18, display: 'flex', alignItems: 'flex-start', gap: 8,
                    }}
                  >
                    <span style={{ flexShrink: 0, marginTop: 1 }}>⚠</span>
                    <span>{error}</span>
                  </motion.div>
                )}
              </AnimatePresence>

              <motion.button
                type="button"
                onClick={handleSubmit}
                disabled={loading}
                whileHover={!loading ? { y: -2, boxShadow: '0 8px 28px rgba(99,102,241,0.45)' } : {}}
                whileTap={!loading ? { scale: 0.97 } : {}}
                style={{
                  width: '100%', padding: '14px 0',
                  background: loading ? 'var(--surface2)' : 'linear-gradient(135deg, var(--accent), var(--accent2))',
                  color: loading ? 'var(--muted)' : '#fff',
                  fontWeight: 700, fontSize: 14,
                  border: loading ? '1px solid var(--border)' : 'none',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                  letterSpacing: '0.02em', borderRadius: 10,
                  position: 'relative', overflow: 'hidden', cursor: loading ? 'not-allowed' : 'pointer',
                }}
              >
                {!loading && (
                  <motion.div
                    initial={{ x: '-100%', opacity: 0 }}
                    whileHover={{ x: '100%', opacity: 0.15 }}
                    transition={{ duration: 0.5 }}
                    style={{
                      position: 'absolute', inset: 0,
                      background: 'linear-gradient(90deg, transparent, white, transparent)',
                      pointerEvents: 'none',
                    }}
                  />
                )}
                {loading ? (
                  <>
                    <motion.div animate={{ rotate: 360 }} transition={{ duration: 0.8, repeat: Infinity, ease: 'linear' }}>
                      <Loader size={15} />
                    </motion.div>
                    {mode === 'signin' ? 'Signing in…' : 'Creating account…'}
                  </>
                ) : (
                  <>
                    {mode === 'signin' ? <LogIn size={15} /> : <UserPlus size={15} />}
                    {mode === 'signin' ? 'Sign In' : 'Create Account'}
                    <ArrowRight size={14} />
                  </>
                )}
              </motion.button>
            </motion.div>
          </div>

          {/* Google OAuth button */}
          <div style={{ margin: '16px 0', display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
            <span style={{ fontSize: 11, color: 'var(--muted)', fontWeight: 500 }}>or</span>
            <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
          </div>

          <motion.button
            type="button"
            whileHover={{ scale: 1.01, boxShadow: '0 4px 16px rgba(0,0,0,0.15)' }}
            whileTap={{ scale: 0.98 }}
            onClick={() => {
              alert('Google OAuth — coming soon. Use username/password login for now.')
            }}
            style={{
              width: '100%',
              padding: '11px 0',
              background: 'var(--surface2)',
              border: '1px solid var(--border)',
              borderRadius: 10,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 10,
              fontSize: 13,
              fontWeight: 600,
              color: 'var(--text)',
              cursor: 'pointer',
            }}
          >
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
              <path d="M17.64 9.205c0-.639-.057-1.252-.164-1.841H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.259h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
              <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
              <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
              <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
            </svg>
            Continue with Google
          </motion.button>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6, duration: 0.4 }}
            style={{ textAlign: 'center', marginTop: 28, fontSize: 11, color: 'var(--muted)', lineHeight: 1.6 }}
          >
            Velox · Multi-Cloud Workload Scheduler
            <br />
            PPO · SHAP · Kafka · Kubernetes
          </motion.div>
        </motion.div>
      </motion.main>
    </div>
  )
}