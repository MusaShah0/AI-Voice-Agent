from .models import Appointment, Task, Transcript, Doctor

class DataRepository:

    @staticmethod
    def save_transcript(session_id, full_transcript):
        return Transcript.objects.create(
            session_id=session_id,
            full_transcript=full_transcript,
        )
    
    @staticmethod
    def create_appointment_sync(caller_name, start_time, end_time, purpose):
        doctor = Doctor.objects.get(pk=Doctor.SINGLETON_ID)
        appointment = Appointment.objects.create(
            doctor=doctor,
            caller_name=caller_name,
            start_time=start_time,
            end_time=end_time,
            purpose=purpose
        )
        return appointment.id

    @staticmethod
    def create_task_sync(appointment_id, description, due_time):
        task = Task.objects.create(
            doctor_id=Doctor.SINGLETON_ID,
            appointment_id=appointment_id,
            description=description,
            due_time=due_time
        )
        return task.id