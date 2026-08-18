# SAM3 prompt lab prototype

This is the old Flask/SAM3 proof of concept, isolated from the production
FastAPI application so it does not conflict with `app/backend/main.py`.

It needs the local checkpoint at `models/sam3/sam3.pt`, the CUDA SAM3
environment, and the sample catalog at `app/backend/data/shots.json`.

```bash
/home/long/.venvs/sam3/bin/python backend/app.py
cd frontend && npm install && npm run dev
```

Open `http://localhost:5174`. Runtime uploads and rendered masks are stored in
`backend/runtime/`, which is ignored by Git.
