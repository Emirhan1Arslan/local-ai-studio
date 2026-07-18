// Local AI Studio kabuğu: sidecar (studio-api) yaşam döngüsü (ADR-0002).
// Başlat → stdout'tan {"port","token"} el sıkışmasını oku → çıkışta süreci öldür.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use serde::Serialize;
use tauri::{Manager, RunEvent, State};

#[derive(Clone, Serialize, Default)]
struct Backend {
    port: u16,
    token: String,
}

struct Sidecar {
    child: Mutex<Option<Child>>,
    backend: Mutex<Backend>,
}

/// UI, backend adresini ve token'ını bu komutla öğrenir.
#[tauri::command]
fn backend_info(state: State<Sidecar>) -> Backend {
    state.backend.lock().unwrap().clone()
}

fn spawn_backend() -> std::io::Result<(Child, Backend)> {
    // Dev modu: uv workspace kökünden çalıştırılır (cwd = src-tauri → ../../.. = repo kökü).
    // Paketli sürümde gömülü python-build-standalone'a geçilecek (ADR-0013, Faz 15).
    let mut child = Command::new("uv")
        .args(["run", "studio-api", "--port", "0", "--token", "auto"])
        .current_dir("../../..")
        .stdout(Stdio::piped())
        .spawn()?;

    let stdout = child.stdout.take().expect("sidecar stdout alınamadı");
    let mut line = String::new();
    BufReader::new(stdout).read_line(&mut line)?;
    let v: serde_json::Value = serde_json::from_str(line.trim()).unwrap_or_default();
    let backend = Backend {
        port: v["port"].as_u64().unwrap_or(0) as u16,
        token: v["token"].as_str().unwrap_or_default().to_string(),
    };
    Ok((child, backend))
}

fn main() {
    tauri::Builder::default()
        .manage(Sidecar {
            child: Mutex::new(None),
            backend: Mutex::new(Backend::default()),
        })
        .invoke_handler(tauri::generate_handler![backend_info])
        .setup(|app| {
            let state: State<Sidecar> = app.state();
            match spawn_backend() {
                Ok((child, backend)) => {
                    *state.child.lock().unwrap() = Some(child);
                    *state.backend.lock().unwrap() = backend;
                }
                Err(e) => eprintln!("sidecar başlatılamadı: {e}"),
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("tauri başlatılamadı")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Süreç sızıntısı olmaması Faz 1 DoD'sidir.
                let state: State<Sidecar> = app.state();
                let child = state.child.lock().unwrap().take();
                if let Some(mut child) = child {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
