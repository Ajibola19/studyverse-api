# Gunicorn configuration for StudyVerse API.
# Long MCQ jobs run in a background thread and may take several minutes.

bind = "0.0.0.0:10000"
workers = 1
worker_class = "gthread"
threads = 4
timeout = 300
graceful_timeout = 30
keepalive = 5
