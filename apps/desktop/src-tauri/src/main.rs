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
    let mut reader = BufReader::new(stdout);
    let mut backend = Backend::default();
    let mut line = String::new();
    // uv/araç gürültüsüne karşı dayanıklılık: ilk 20 satırda {"port",...} JSON'ını ara.
    for _ in 0..20 {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break; // sidecar erken öldü
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) {
            if let Some(port) = v["port"].as_u64() {
                backend.port = port as u16;
                backend.token = v["token"].as_str().unwrap_or_default().to_string();
                break;
            }
        }
    }
    // Pipe'ı açık tutup boşalt: kapatırsak sidecar'ın sonraki stdout yazımı patlar,
    // boşaltmazsak tampon dolunca sidecar bloklanır.
    std::thread::spawn(move || {
        let mut sink = String::new();
        loop {
            sink.clear();
            match reader.read_line(&mut sink) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
    });
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
                // Süreç sızıntısı olmaması Faz 1 DoD'sidir. `uv run` bir süreç AĞACI
                // doğurur (uv → venv python → sunucu); child.kill() yalnız kökü öldürür
                // ve sunucu yetim kalırdı (canlı testte yakalandı) → ağacı komple öldür.
                let state: State<Sidecar> = app.state();
                let child = state.child.lock().unwrap().take();
                if let Some(mut child) = child {
                    #[cfg(windows)]
                    {
                        let _ = Command::new("taskkill")
                            .args(["/PID", &child.id().to_string(), "/T", "/F"])
                            .status();
                    }
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}
