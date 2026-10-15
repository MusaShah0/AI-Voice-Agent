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
import DoctorCallPanel from './DoctorCallPanel'
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
  const [activeReminders, setActiveReminders] = useState([])
  const [activeCall, setActiveCall] = useState(null)
  const [reminderError, setReminderError] = useState('')
  const [startingCallFor, setStartingCallFor] = useState(null)

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

  const loadActiveReminders = useCallback(async (signal) => {
    try {
      const response = await fetch(`${API_URL}/api/doc/reminders/active/`, {
        headers: { Accept: 'application/json' },
        signal,
      })
      if (!response.ok) throw new Error(`Reminder refresh failed (${response.status}).`)
      const data = await response.json()
      setActiveReminders(data.active || [])
      setReminderError('')
    } catch (requestError) {
      if (requestError.name !== 'AbortError') {
        setReminderError(requestError.message || 'Could not refresh active reminders.')
      }
    }
  }, [])

  const startReminderCall = async (reminder) => {
    setReminderError('')
    setStartingCallFor(reminder.reminder_id)
    let stream
    let audioContext
    let cancelled = false
    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Microphone access requires localhost or a secure browser context.')
      }
      const AudioContextClass = window.AudioContext || window.webkitAudioContext
      if (!AudioContextClass) throw new Error('This browser does not support audio calls.')
      audioContext = new AudioContextClass({ sampleRate: 16000 })
      const resumePromise = audioContext.resume()
      const streamPromise = navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: false,
          autoGainControl: true,
        },
      }).then((acquiredStream) => {
        if (cancelled) {
          acquiredStream.getTracks().forEach((track) => track.stop())
        } else {
          stream = acquiredStream
        }
        return acquiredStream
      })
      const responsePromise = fetch(
        `${API_URL}/api/doc/reminders/${reminder.reminder_id}/call-token/`,
        { method: 'POST', headers: { Accept: 'application/json' } },
      )
      const [, , response] = await Promise.all([
        resumePromise,
        streamPromise,
        responsePromise,
      ])
      const result = await response.json()
      if (!response.ok) throw new Error(result.error || `Could not start call (${response.status}).`)
      setActiveCall({
        reminder,
        ws_url: result.ws_url,
        stream,
        audioContext,
      })
      stream = null
      audioContext = null
    } catch (requestError) {
      cancelled = true
      stream?.getTracks().forEach((track) => track.stop())
      if (audioContext && audioContext.state !== 'closed') await audioContext.close()
      setReminderError(requestError.message || 'Could not start the reminder call.')
    } finally {
      setStartingCallFor(null)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    loadDashboard(controller.signal)
    loadActiveReminders(controller.signal)
    const intervalId = window.setInterval(() => {
      loadDashboard()
    }, 60_000)
    const reminderIntervalId = window.setInterval(() => {
      loadActiveReminders()
    }, 10_000)
    return () => {
      controller.abort()
      window.clearInterval(intervalId)
      window.clearInterval(reminderIntervalId)
    }
  }, [loadDashboard, loadActiveReminders])

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

          {activeReminders.length > 0 && (
            <section className="doctor-reminder" role="status" aria-live="polite">
              <div className="doctor-reminder-icon"><Bell size={19} /></div>
              <div>
                <strong>
                  {activeReminders.length === 1
                    ? 'Appointment reminder'
                    : `${activeReminders.length} active appointment reminders`}
                </strong>
                <div className="doctor-reminder-list">
                  {activeReminders.map((reminder) => (
                    <div className="doctor-reminder-item" key={reminder.reminder_id}>
                      <div>
                        <p>
                          <strong>{reminder.patient_name}</strong>
                          {' · '}
                          {formatTime(reminder.start_time, timeZone)}
                          {' · '}
                          {reminder.minutes_until_start <= 0
                            ? 'starting now'
                            : `${reminder.minutes_until_start} min`}
                        </p>
                        <span>
                          Reminder active
                          {reminder.status === 'snoozed' && reminder.next_fire_at
                            ? ` · Snoozed until ${formatTime(reminder.next_fire_at, timeZone)}`
                            : ` · Attempt ${reminder.attempt_count}`}
                        </span>
                      </div>
                      <button
                        type="button"
                        className="doctor-reminder-call"
                        disabled={Boolean(startingCallFor || activeCall)}
                        onClick={() => startReminderCall(reminder)}
                      >
                        {startingCallFor === reminder.reminder_id ? 'Connecting…' : 'Call'}
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </section>
          )}
          {reminderError && <div className="doctor-error" role="alert"><CircleAlert size={17} /> {reminderError}</div>}
          {activeCall && <DoctorCallPanel call={activeCall} onClose={() => setActiveCall(null)} />}

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
