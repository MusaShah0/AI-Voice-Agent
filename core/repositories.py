from datetime import timedelta

from django.db import transaction

from .models import Appointment, Task, Transcript, Doctor


class DataRepository:

    @staticmethod
    def list_upcoming_appointments_sync(caller_name, now):
        return list(
            Appointment.objects.filter(
                caller_name__iexact=caller_name.strip(),
                status="Scheduled",
                start_time__gte=now,
            ).order_by("start_time").values(
                "id",
                "caller_name",
                "start_time",
                "end_time",
                "purpose",
            )[:10]
        )

    @staticmethod
    def save_transcript(session_id, full_transcript):
        return Transcript.objects.create(
            session_id=session_id,
            full_transcript=full_transcript,
        )
    
    @staticmethod
    def create_appointment_sync(
        caller_name,
        start_time,
        end_time,
        purpose,
        google_event_id=None,
    ):
        doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
        appointment = Appointment.objects.create(
            doctor=doctor,
            caller_name=caller_name,
            start_time=start_time,
            end_time=end_time,
            purpose=purpose,
            google_event_id=google_event_id,
        )
        return appointment.id

    @staticmethod
    def update_appointment_sync(appointment_id, start_time, end_time):
        with transaction.atomic():
            appointment = Appointment.objects.select_for_update().get(
                pk=appointment_id,
                status="Scheduled",
            )
            conflict = Appointment.objects.filter(
                status="Scheduled",
                start_time__lt=end_time,
                end_time__gt=start_time,
            ).exclude(pk=appointment.pk).first()
            if conflict:
                return {"status": "conflict", "appointment_id": conflict.pk}

            appointment.start_time = start_time
            appointment.end_time = end_time
            appointment.reminder_sent_at = None
            appointment.save(update_fields=["start_time", "end_time", "reminder_sent_at"])
            Task.objects.filter(
                appointment=appointment,
                status="Pending",
            ).update(due_time=start_time - timedelta(hours=1))
            return {"status": "updated"}

    @staticmethod
    def cancel_appointment_sync(appointment_id):
        with transaction.atomic():
            appointment = Appointment.objects.select_for_update().get(
                pk=appointment_id,
                status="Scheduled",
            )
            appointment.status = "Cancelled"
            appointment.save(update_fields=["status"])
            Task.objects.filter(
                appointment=appointment,
                status="Pending",
            ).update(status="Cancelled")
            return {"status": "cancelled"}

    @staticmethod
    def create_task_sync(appointment_id, description, due_time):
        task = Task.objects.create(
            doctor_id=Doctor.SINGLETON_ID,
            appointment_id=appointment_id,
            description=description,
            due_time=due_time
        )
        return task.id