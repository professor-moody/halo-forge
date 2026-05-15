use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, WindowEvent};
use tauri_plugin_shell::{process::CommandChild, ShellExt};

struct SidecarState(Mutex<Option<CommandChild>>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(SidecarState(Mutex::new(None)))
        .setup(|app| {
            let sidecar = app
                .shell()
                .sidecar("halo-forge-runtime")
                .expect("halo-forge-runtime sidecar must be bundled");
            let (_rx, child) = sidecar
                .args(["serve-public", "--host", "127.0.0.1", "--port", "8000"])
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
    let Ok(mut stream) = TcpStream::connect(("127.0.0.1", 8000)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
    let request = b"GET /api/public/health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n";
    if stream.write_all(request).is_err() {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200")
}
