# Remote Dev Guide (Docker + SSH)

This project runs inside a shared Docker container. An external port is already forwarded:
- **External 20225 -> Container 5173 (Vite)**

## Install & Run
From the repo root:

```bash
cd frontend
npm install
npm run dev:remote
```

Access the app at:

```
http://<SERVER_IP_OR_DOMAIN>:20225
```

## Stop the Server
- Foreground: press `Ctrl+C` in the terminal running Vite.
- Background (recommended): use `tmux`.

Example (from repo root):

```bash
cd frontend
tmux new -s workbook-ui
npm run dev:remote
```

Detach with `Ctrl+B`, then `D`. Reattach later with:

```bash
cd frontend
tmux attach -t workbook-ui
```

## Updates & HMR
- HMR should pick up changes automatically when possible.
- If HMR fails (common with remote WebSockets), restart Vite.

Restart quickly:

```bash
cd frontend
npm run dev:remote
```

## Check / Kill Port Usage
From the repo root:

```bash
cd frontend
lsof -i :5173
```

Or:

```bash
cd frontend
ss -lptn 'sport = :5173'
```

Kill by PID (replace `<PID>`):

```bash
cd frontend
kill -9 <PID>
```

## HMR Troubleshooting Checklist
1) Ensure Vite runs on `0.0.0.0:5173`:

```bash
cd frontend
npm run dev:remote
```

2) If live reload fails, set the HMR client port to the external port:

```bash
cd frontend
VITE_HMR_CLIENT_PORT=20225 npm run dev:remote
```

3) Confirm the external URL is correct:

```
http://<SERVER_IP_OR_DOMAIN>:20225
```
