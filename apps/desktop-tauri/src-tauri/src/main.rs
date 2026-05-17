use serde::Serialize;
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager, RunEvent, State, WindowEvent};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

const DASHBOARD_HOST: &str = "127.0.0.1";
const DASHBOARD_PORT: u16 = 8765;
const DASHBOARD_URL: &str = "http://127.0.0.1:8765";
const RELEASE_CHANNEL: &str = "alpha";

#[derive(Clone, Serialize)]
struct DesktopStatus {
    state: String,
    message: String,
    detail: Option<String>,
    log_path: Option<String>,
    dashboard_url: String,
    app_version: String,
    release_channel: String,
}

impl DesktopStatus {
    fn new(state: &str, message: &str, detail: Option<String>, log_path: Option<PathBuf>) -> Self {
        Self {
            state: state.to_string(),
            message: message.to_string(),
            detail,
            log_path: log_path.map(|path| path.display().to_string()),
            dashboard_url: DASHBOARD_URL.to_string(),
            app_version: env!("CARGO_PKG_VERSION").to_string(),
            release_channel: RELEASE_CHANNEL.to_string(),
        }
    }
}

struct RuntimeState {
    child: Option<CommandChild>,
    starting: bool,
    status: DesktopStatus,
}

struct DesktopState(Mutex<RuntimeState>);

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(DesktopState(Mutex::new(RuntimeState {
            child: None,
            starting: false,
            status: DesktopStatus::new(
                "starting",
                "Starting Halo Forge.",
                Some("Preparing the local dashboard runtime.".to_string()),
                None,
            ),
        })))
        .invoke_handler(tauri::generate_handler![desktop_status, desktop_retry])
        .setup(|app| {
            start_runtime(app.handle().clone());
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                stop_owned_sidecar(window.app_handle());
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building Halo Forge desktop app");

    app.run(|app_handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. } | RunEvent::Exit) {
            stop_owned_sidecar(app_handle);
        }
    });
}

fn main() {
    run();
}

#[tauri::command]
fn desktop_status(state: State<'_, DesktopState>) -> DesktopStatus {
    state
        .0
        .lock()
        .expect("desktop state lock poisoned")
        .status
        .clone()
}

#[tauri::command]
fn desktop_retry(app: AppHandle) -> DesktopStatus {
    stop_owned_sidecar(&app);
    start_runtime(app.clone());
    app.state::<DesktopState>()
        .0
        .lock()
        .expect("desktop state lock poisoned")
        .status
        .clone()
}

fn start_runtime(app: AppHandle) {
    {
        let state = app.state::<DesktopState>();
        let mut guard = state.0.lock().expect("desktop state lock poisoned");
        if guard.starting {
            return;
        }
        guard.starting = true;
        guard.status = DesktopStatus::new(
            "starting",
            "Starting Halo Forge.",
            Some("Checking the desktop runtime and local dashboard port.".to_string()),
            Some(desktop_log_path()),
        );
    }

    thread::spawn(move || {
        let log_path = desktop_log_path();
        if let Err(err) = fs::create_dir_all(log_path.parent().unwrap_or_else(|| ".".as_ref())) {
            set_status(
                &app,
                "runtime_missing",
                "Could not prepare the desktop runtime log.",
                Some(err.to_string()),
                Some(log_path),
            );
            mark_not_starting(&app);
            return;
        }

        if let Err(err) = validate_runtime_inputs(&app) {
            set_status(
                &app,
                err.state,
                err.message,
                Some(err.detail),
                Some(log_path),
            );
            mark_not_starting(&app);
            return;
        }

        if port_is_in_use(DASHBOARD_HOST, DASHBOARD_PORT) {
            if health_ok() {
                set_status(
                    &app,
                    "ready",
                    "Halo Forge is already running.",
                    Some(
                        "Using the local dashboard that is already listening on 127.0.0.1:8765."
                            .to_string(),
                    ),
                    Some(log_path),
                );
            } else {
                set_status(
                    &app,
                    "port_conflict",
                    "Port 8765 is already in use.",
                    Some(
                        "Close the process using 127.0.0.1:8765, then retry Halo Forge."
                            .to_string(),
                    ),
                    Some(log_path),
                );
            }
            mark_not_starting(&app);
            return;
        }

        let frontend_dist = match app.path().resource_dir() {
            Ok(path) => path.join("frontend"),
            Err(err) => {
                set_status(
                    &app,
                    "runtime_missing",
                    "Could not locate bundled desktop resources.",
                    Some(err.to_string()),
                    Some(log_path),
                );
                mark_not_starting(&app);
                return;
            }
        };
        let repo_root = dev_repo_root();
        let bundled_runtime = bundled_runtime_dir_if_available(&app);
        let sidecar = match app.shell().sidecar("halo-forge-runtime") {
            Ok(sidecar) => sidecar,
            Err(err) => {
                set_status(
                    &app,
                    "runtime_missing",
                    "Halo Forge runtime sidecar is missing.",
                    Some(err.to_string()),
                    Some(log_path),
                );
                mark_not_starting(&app);
                return;
            }
        };

        let mut sidecar_command = sidecar
            .env("HALO_FORGE_FRONTEND_DIST", frontend_dist)
            .env("HALO_FORGE_REPO_ROOT", repo_root);
        if let Some(runtime_dir) = bundled_runtime {
            sidecar_command = sidecar_command.env("HALO_FORGE_BUNDLED_RUNTIME", runtime_dir);
        }
        let spawn_result = sidecar_command
            .args([
                "dashboard",
                "--no-build",
                "--host",
                DASHBOARD_HOST,
                "--port",
                &DASHBOARD_PORT.to_string(),
            ])
            .spawn();

        let (mut rx, child) = match spawn_result {
            Ok(result) => result,
            Err(err) => {
                set_status(
                    &app,
                    "runtime_missing",
                    "Could not start the Halo Forge runtime.",
                    Some(err.to_string()),
                    Some(log_path),
                );
                mark_not_starting(&app);
                return;
            }
        };

        {
            let state = app.state::<DesktopState>();
            let mut guard = state.0.lock().expect("desktop state lock poisoned");
            guard.child = Some(child);
            guard.status = DesktopStatus::new(
                "starting",
                "Starting the dashboard service.",
                Some("Waiting for /api/public/health to report ready.".to_string()),
                Some(log_path.clone()),
            );
        }

        let log_app = app.clone();
        let log_path_for_thread = log_path.clone();
        thread::spawn(move || {
            tauri::async_runtime::block_on(async move {
                while let Some(event) = rx.recv().await {
                    record_sidecar_event(&log_app, &log_path_for_thread, event);
                }
            });
        });

        let deadline = Instant::now() + Duration::from_secs(30);
        while Instant::now() < deadline {
            if health_ok() {
                set_status(
                    &app,
                    "ready",
                    "Halo Forge is ready.",
                    Some("Opening the local dashboard.".to_string()),
                    Some(log_path),
                );
                mark_not_starting(&app);
                return;
            }
            thread::sleep(Duration::from_millis(250));
        }

        set_status(
            &app,
            "health_timeout",
            "Halo Forge did not become ready in time.",
            Some("The runtime started, but /api/public/health did not answer within 30 seconds. Check the log, then retry.".to_string()),
            Some(log_path),
        );
        mark_not_starting(&app);
    });
}

fn validate_runtime_inputs(app: &AppHandle) -> Result<(), StartupError> {
    let frontend_dist = app
        .path()
        .resource_dir()
        .map_err(|err| {
            StartupError::new(
                "runtime_missing",
                "Bundled desktop resources are unavailable.",
                err.to_string(),
            )
        })?
        .join("frontend");
    if !frontend_dist.join("index.html").is_file() {
        return Err(StartupError::new(
            "runtime_missing",
            "Bundled dashboard assets are missing.",
            format!("Expected {}", frontend_dist.join("index.html").display()),
        ));
    }

    let bundled_runtime = bundled_runtime_executable(app);
    if bundled_runtime.is_file() {
        run_bundled_self_check(&bundled_runtime, &frontend_dist)?;
        return Ok(());
    }

    let python = dev_repo_root().join(".venv/bin/python");
    if python.is_file() {
        return Ok(());
    }

    Err(StartupError::new(
        "runtime_missing",
        "Bundled Halo Forge runtime is missing.",
        format!(
            "Expected bundled runtime at {} or source dev runtime at {}",
            bundled_runtime.display(),
            python.display()
        ),
    ))
}

struct StartupError {
    state: &'static str,
    message: &'static str,
    detail: String,
}

impl StartupError {
    fn new(state: &'static str, message: &'static str, detail: String) -> Self {
        Self {
            state,
            message,
            detail,
        }
    }
}

fn record_sidecar_event(app: &AppHandle, log_path: &PathBuf, event: CommandEvent) {
    let line = match event {
        CommandEvent::Stdout(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        CommandEvent::Stderr(bytes) => String::from_utf8_lossy(&bytes).to_string(),
        CommandEvent::Error(err) => format!("sidecar error: {err}"),
        CommandEvent::Terminated(payload) => {
            let detail = format!(
                "Runtime exited with code {:?} signal {:?}.",
                payload.code, payload.signal
            );
            clear_child(app);
            if health_ok() {
                append_log(
                    log_path,
                    "Runtime parent exited while the dashboard still answers; stopping owned dashboard listener.",
                );
                stop_dashboard_listener();
            }
            set_status(
                app,
                "backend_exited",
                "Halo Forge runtime exited.",
                Some(detail.clone()),
                Some(log_path.clone()),
            );
            detail
        }
        _ => "sidecar event".to_string(),
    };
    append_log(log_path, &line);
}

fn append_log(path: &PathBuf, line: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{}", line.trim_end());
    }
}

fn set_status(
    app: &AppHandle,
    state_name: &str,
    message: &str,
    detail: Option<String>,
    log_path: Option<PathBuf>,
) {
    let state = app.state::<DesktopState>();
    let mut guard = state.0.lock().expect("desktop state lock poisoned");
    guard.status = DesktopStatus::new(state_name, message, detail, log_path);
}

fn mark_not_starting(app: &AppHandle) {
    let state = app.state::<DesktopState>();
    let mut guard = state.0.lock().expect("desktop state lock poisoned");
    guard.starting = false;
}

fn clear_child(app: &AppHandle) {
    let state = app.state::<DesktopState>();
    let mut guard = state.0.lock().expect("desktop state lock poisoned");
    guard.child = None;
    guard.starting = false;
}

fn stop_owned_sidecar(app: &AppHandle) {
    let child = {
        let state = app.state::<DesktopState>();
        let mut guard = state.0.lock().expect("desktop state lock poisoned");
        guard.child.take()
    };
    if let Some(child) = child {
        let pid = child.pid();
        stop_descendant_processes(pid);
        let _ = child.kill();
        if wait_for_dashboard_to_stop(Duration::from_secs(2)) {
            return;
        }
        stop_dashboard_listener();
        let _ = wait_for_dashboard_to_stop(Duration::from_secs(2));
    }
}

#[cfg(unix)]
fn stop_descendant_processes(root_pid: u32) {
    let mut descendants = collect_descendant_pids(root_pid);
    descendants.reverse();
    signal_pids(&descendants, "TERM");
    thread::sleep(Duration::from_millis(300));
    signal_pids(&descendants, "KILL");
}

#[cfg(not(unix))]
fn stop_descendant_processes(_root_pid: u32) {}

#[cfg(unix)]
fn collect_descendant_pids(parent_pid: u32) -> Vec<u32> {
    let mut descendants = Vec::new();
    let Ok(output) = StdCommand::new("pgrep")
        .arg("-P")
        .arg(parent_pid.to_string())
        .output()
    else {
        return descendants;
    };
    if !output.status.success() {
        return descendants;
    }
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        if let Ok(pid) = line.trim().parse::<u32>() {
            descendants.push(pid);
            descendants.extend(collect_descendant_pids(pid));
        }
    }
    descendants
}

#[cfg(unix)]
fn signal_pids(pids: &[u32], signal: &str) {
    if pids.is_empty() {
        return;
    }
    let mut command = StdCommand::new("kill");
    command.arg(format!("-{signal}"));
    for pid in pids {
        command.arg(pid.to_string());
    }
    let _ = command.status();
}

fn wait_for_dashboard_to_stop(timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if !port_is_in_use(DASHBOARD_HOST, DASHBOARD_PORT) {
            return true;
        }
        thread::sleep(Duration::from_millis(100));
    }
    !port_is_in_use(DASHBOARD_HOST, DASHBOARD_PORT)
}

#[cfg(unix)]
fn stop_dashboard_listener() {
    let mut pids = dashboard_listener_pids();
    pids.sort_unstable();
    pids.dedup();
    signal_pids(&pids, "TERM");
    thread::sleep(Duration::from_millis(300));
    if port_is_in_use(DASHBOARD_HOST, DASHBOARD_PORT) {
        signal_pids(&pids, "KILL");
    }
}

#[cfg(not(unix))]
fn stop_dashboard_listener() {}

#[cfg(unix)]
fn dashboard_listener_pids() -> Vec<u32> {
    let lsof_query = format!("TCP:{DASHBOARD_PORT}");
    let Ok(output) = StdCommand::new("lsof")
        .args(["-nP", "-t", "-i", &lsof_query, "-sTCP:LISTEN"])
        .output()
    else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.trim().parse::<u32>().ok())
        .collect()
}

fn dev_repo_root() -> PathBuf {
    std::env::var_os("HALO_FORGE_REPO_ROOT")
        .map(PathBuf::from)
        .or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .canonicalize()
                .ok()
        })
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn bundled_runtime_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path()
        .resource_dir()
        .ok()
        .map(|path| path.join("runtime/halo-forge-runtime"))
}

fn bundled_runtime_dir_if_available(app: &AppHandle) -> Option<PathBuf> {
    let dir = bundled_runtime_dir(app)?;
    if runtime_executable_in_dir(&dir).is_file() {
        Some(dir)
    } else {
        None
    }
}

fn bundled_runtime_executable(app: &AppHandle) -> PathBuf {
    bundled_runtime_dir(app)
        .map(|dir| runtime_executable_in_dir(&dir))
        .unwrap_or_else(|| PathBuf::from("runtime/halo-forge-runtime/halo-forge-runtime"))
}

fn runtime_executable_in_dir(dir: &PathBuf) -> PathBuf {
    dir.join("halo-forge-runtime")
}

fn run_bundled_self_check(
    runtime_exe: &PathBuf,
    frontend_dist: &PathBuf,
) -> Result<(), StartupError> {
    let output = StdCommand::new(runtime_exe)
        .arg("--desktop-self-check")
        .env("HALO_FORGE_FRONTEND_DIST", frontend_dist)
        .output()
        .map_err(|err| {
            StartupError::new(
                "runtime_self_check_failed",
                "Could not check the bundled Halo Forge runtime.",
                err.to_string(),
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(StartupError::new(
        "runtime_self_check_failed",
        "Bundled Halo Forge runtime failed its self-check.",
        format!("stdout: {}\nstderr: {}", stdout.trim(), stderr.trim()),
    ))
}

fn desktop_log_path() -> PathBuf {
    std::env::var_os("HALO_FORGE_DESKTOP_LOG")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".halo-forge/desktop/runtime.log")
        })
}

fn port_is_in_use(host: &str, port: u16) -> bool {
    TcpStream::connect((host, port)).is_ok()
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
