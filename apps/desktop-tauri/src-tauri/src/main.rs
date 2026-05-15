use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, WindowEvent};
use tauri_plugin_shell::{process::CommandChild, ShellExt};

const DASHBOARD_HOST: &str = "127.0.0.1";
const DASHBOARD_PORT: u16 = 8765;

struct SidecarState(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .setup(|app| {
            let frontend_dist = app.path().resource_dir()?.join("frontend");
            let repo_root = dev_repo_root();
            let sidecar = app
                .shell()
                .sidecar("halo-forge-runtime")
                .expect("halo-forge-runtime sidecar must be bundled");
            let (_rx, child) = sidecar
                .env("HALO_FORGE_FRONTEND_DIST", frontend_dist)
                .env("HALO_FORGE_REPO_ROOT", repo_root)
                .args([
                    "dashboard",
                    "--no-build",
                    "--host",
                    DASHBOARD_HOST,
                    "--port",
                    &DASHBOARD_PORT.to_string(),
                ])
                .spawn()
                .expect("failed to start Halo Forge runtime sidecar");
            let state = app.state::<SidecarState>();
            *state.0.lock().expect("sidecar state lock poisoned") = Some(child);
            wait_for_dashboard_health();
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                let state = window.state::<SidecarState>();
                let child = {
                    let mut guard = state.0.lock().expect("sidecar state lock poisoned");
                    guard.take()
                };
                if let Some(child) = child {
                    let _ = child.kill();
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running Halo Forge desktop app");
}

fn main() {
    run();
}

fn dev_repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn wait_for_dashboard_health() {
    let deadline = Instant::now() + Duration::from_secs(30);
    while Instant::now() < deadline {
        if health_ok() {
            return;
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn health_ok() -> bool {
    let Ok(mut stream) = TcpStream::connect((DASHBOARD_HOST, DASHBOARD_PORT)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request =
        b"GET /api/public/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200")
}
