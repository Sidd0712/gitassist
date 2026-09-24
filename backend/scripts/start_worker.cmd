@echo off
rem Starts the GitAssist embedding + indexing worker (see local_embed_worker.py).
rem Run at login with Task Scheduler:
rem   schtasks /Create /TN "GitAssist worker" /SC ONLOGON /TR "\"%~f0\""
cd /d "%~dp0.."
python scripts\local_embed_worker.py
