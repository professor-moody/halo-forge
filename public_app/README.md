# halo-forge Public Frontend

User-facing Next.js app for:

- training launch
- live run monitoring
- results and guided recovery
- modality readiness
- in-app docs summaries

## Run locally

Start the public API:

```bash
uvicorn halo_forge.public_api.app:app --host 127.0.0.1 --port 8081
```

Then start the frontend:

```bash
cd public_app
npm install
npm run dev
```

Set `NEXT_PUBLIC_HALO_API_BASE` if the API is not on `http://127.0.0.1:8081/api/public`.
