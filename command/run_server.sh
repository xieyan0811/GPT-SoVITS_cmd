if [ $# -gt 0 ]; then
    w=$1
else
    w=2
fi
export TTS_WORKERS=$w
echo uvicorn command.serv:app --host 0.0.0.0 --port 9880 --workers $w
uvicorn command.serv:app --host 0.0.0.0 --port 9880 --workers $w
