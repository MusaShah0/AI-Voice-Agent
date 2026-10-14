import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  Bell,
  CalendarClock,
  CalendarDays,
  CircleAlert,
  Clock3,
  RefreshCw,
  Search,
  UserRound,
} from 'lucide-react'
import './DoctorDashboard.css'

const API_URL = import.meta.env.VITE_API_BASE_URL || ''

function formatDateTime(value, timeZone) {
  return new Intl.DateTimeFormat('en-PK', {
    timeZone,
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
}

function formatTime(value, timeZone) {
  return new Intl.DateTimeFormat('en-PK', {
    timeZone,
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
}

function AppointmentCard({ appointment, timeZone }) {
  return (
    <article className="doctor-appointment-card">
      <div className="doctor-card-date">
        <CalendarClock size={17} />
        <span>{formatDateTime(appointment.start_time, timeZone)}</span>
      </div>
      <div className="doctor-card-person">
        <div className="doctor-avatar"><UserRound size={18} /></div>
        <div>
          <h3>{appointment.patient_name}</h3>
          <span>
            {formatTime(appointment.start_time, timeZone)}
            {' – '}
            {formatTime(appointment.end_time, timeZone)}
          </span>
        </div>
        <span className="doctor-status scheduled">{appointment.status}</span>
      </div>
      <div className="doctor-purpose">
        <span>VISIT PURPOSE</span>
        <p>{appointment.purpose}</p>
      </div>
    </article>
  )
}

function DoctorDashboard() {
  const [dashboard, setDashboard] = useState(null)
  const [activeTab, setActiveTab] = useState('upcoming')
  const [search, setSearch] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [lastUpdated, setLastUpdated] = useState(null)

  const loadDashboard = useCallback(async (signal) => {
    try {
      const response = await fetch(`${API_URL}/api/doc/dashboard/`, {
        headers: { Accept: 'application/json' },
        signal,
      })
      if (!response.ok) {
        let message = `Dashboard request failed (${response.status}).`
        try {
          const body = await response.json()
          message = body.error || message
        } catch {
          // Keep the HTTP status message when the response body is not JSON.
        }
        throw new Error(message)
      }
      const result = await response.json()
      setDashboard(result)
      setLastUpdated(new Date())
      setError('')
    } catch (requestError) {
      if (requestError.name !== 'AbortError') {
        setError(requestError.message || 'Could not load appointments.')
      }
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    loadDashboard(controller.signal)
    const intervalId = window.setInterval(() => {
      loadDashboard()
    }, 60_000)
    return () => {
      controller.abort()
      window.clearInterval(intervalId)
    }
  }, [loadDashboard])

  const timeZone = dashboard?.doctor?.timezone || 'Asia/Karachi'
  const filteredAppointments = useMemo(() => {
    const source = activeTab === 'upcoming'
      ? dashboard?.upcoming || []
      : dashboard?.all_appointments || []
    const query = search.trim().toLocaleLowerCase()
    if (!query) return source
    return source.filter((appointment) => [
      appointment.patient_name,
      appointment.purpose,
      appointment.status,
    ].some((value) => value.toLocaleLowerCase().includes(query)))
  }, [activeTab, dashboard, search])

  return (
    <div className="doctor-page">
      <aside className="doctor-rail">
        <div className="doctor-rail-mark"><Activity size={21} /></div>
        <span>CLINIC<br />PORTAL</span>
        <div className="doctor-rail-bottom"><span /> LOCAL DEV</div>
      </aside>

      <div className="doctor-main">
        <header className="doctor-topbar">
          <div>
            <div className="doctor-eyebrow">DR. MELVIN’S OFFICE <span>/</span> DOCTOR PORTAL</div>
            <h1>{dashboard?.doctor?.name || 'Doctor dashboard'}</h1>
          </div>
          <div className="doctor-top-actions">
            <span className="doctor-refresh-label">
              {lastUpdated
                ? `Updated ${lastUpdated.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}`
                : 'Loading appointments'}
            </span>
            <button
              className="doctor-refresh"
              type="button"
              onClick={() => loadDashboard()}
              aria-label="Refresh appointments"
              title="Refresh appointments"
            >
              <RefreshCw size={16} />
            </button>
          </div>
        </header>

        <main className="doctor-content">
          <div className="doctor-page-heading">
            <div>
              <span className="doctor-eyebrow">SCHEDULE OVERVIEW</span>
              <h2>Appointments</h2>
              <p>Review the schedule and prepare for upcoming visits.</p>
            </div>
            <div className="doctor-auto-refresh"><span /> AUTO-REFRESH · 60 SEC</div>
          </div>

          {(dashboard?.reminders?.length > 0) && (
            <section className="doctor-reminder" role="status" aria-live="polite">
              <div className="doctor-reminder-icon"><Bell size={19} /></div>
              <div>
                <strong>
                  {dashboard.reminders.length === 1
                    ? 'Appointment starting within 15 minutes'
                    : `${dashboard.reminders.length} appointments starting within 15 minutes`}
                </strong>
                <p>
                  {dashboard.reminders.map((appointment) => (
                    <span key={appointment.id}>
                      {appointment.patient_name} · {formatTime(appointment.start_time, timeZone)}
                    </span>
                  ))}
                </p>
              </div>
            </section>
          )}

          <nav className="doctor-tabs" aria-label="Appointment views">
            <button
              type="button"
              className={activeTab === 'upcoming' ? 'active' : ''}
              onClick={() => setActiveTab('upcoming')}
            >
              <CalendarClock size={16} /> Upcoming
              <span>{dashboard?.upcoming?.length ?? 0}</span>
            </button>
            <button
              type="button"
              className={activeTab === 'all' ? 'active' : ''}
              onClick={() => setActiveTab('all')}
            >
              <CalendarDays size={16} /> All appointments
              <span>{dashboard?.all_appointments?.length ?? 0}</span>
            </button>
          </nav>

          <div className="doctor-list-toolbar">
            <div className="doctor-list-title">
              <h3>{activeTab === 'upcoming' ? 'Upcoming visits' : 'All appointments'}</h3>
              <span>{filteredAppointments.length} shown</span>
            </div>
            <label className="doctor-search">
              <Search size={16} />
              <input
                type="search"
                value={search}
                onChange={(event) => setSearch(event.target.value)}
                placeholder="Search patient, purpose, status"
                aria-label="Search appointments"
              />
            </label>
          </div>

          {error && (
            <div className="doctor-error" role="alert">
              <CircleAlert size={17} /> {error}
            </div>
          )}

          {loading ? (
            <div className="doctor-empty"><RefreshCw size={18} /> Loading appointments…</div>
          ) : filteredAppointments.length ? (
            <div className="doctor-appointment-list">
              {filteredAppointments.map((appointment) => (
                <AppointmentCard
                  key={appointment.id}
                  appointment={appointment}
                  timeZone={timeZone}
                />
              ))}
            </div>
          ) : (
            <div className="doctor-empty">
              <Clock3 size={20} />
              <strong>
                {search
                  ? 'No appointments match your search.'
                  : activeTab === 'upcoming'
                    ? 'No upcoming appointments.'
                    : 'No appointments have been recorded.'}
              </strong>
            </div>
          )}

          <footer className="doctor-footer">
            <span>Patient-facing scheduler</span>
            <a href="/">Open patient appointment page</a>
            <span>Times shown in {timeZone}</span>
          </footer>
        </main>
      </div>
    </div>
  )
}

export default DoctorDashboard
