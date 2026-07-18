// Local AI Studio kabuğu: sidecar (studio-api) yaşam döngüsü (ADR-0002).
// Başlat → stdout'tan {"port","token"} el sıkışmasını oku → çıkışta süreci öldür.
// Beklenmedik çökmede otomatik yeniden başlatma + UI'a "backend-changed" olayı (Faz 1 cila).
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

#[derive(Clone, Serialize, Default)]
struct Backend {
    port: u16,
    token: String,
}

struct Sidecar {
    child: Mutex<Option<Child>>,
    backend: Mutex<Backend>,
    shutting_down: AtomicBool,
}

/// UI, backend adresini ve token'ını bu komutla öğrenir (ilk bağlantı).
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

fn kill_tree(child: &mut Child) {
    // `uv run` bir süreç AĞACI doğurur (uv → venv python → sunucu); kill() yalnız
    // kökü öldürür ve sunucu yetim kalırdı (canlı testte yakalandı) → ağacı komple öldür.
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/PID", &child.id().to_string(), "/T", "/F"])
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

/// (Yeniden) başlatır, state'i günceller, UI'a yeni adresi bildirir.
fn respawn_and_notify(app: &AppHandle, reason: &str) {
    match spawn_backend() {
        Ok((child, backend)) => {
            eprintln!("sidecar {reason}: port={}", backend.port);
            let state: State<Sidecar> = app.state();
            *state.child.lock().unwrap() = Some(child);
            *state.backend.lock().unwrap() = backend.clone();
            let _ = app.emit("backend-changed", &backend);
        }
        Err(e) => eprintln!("sidecar başlatılamadı ({reason}): {e}"),
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Sidecar {
            child: Mutex::new(None),
            backend: Mutex::new(Backend::default()),
            shutting_down: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![backend_info])
        .setup(|app| {
            let handle = app.handle().clone();
            respawn_and_notify(&handle, "başlatıldı");

            // Çökme izleyici: sidecar süreci beklenmedik ölürse (kill -9 dahil) otomatik
            // yeniden doğurur — Faz 1 DoD'sinin "çökmede otomatik yeniden başlatma" kalemi.
            let monitor_handle = app.handle().clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(Duration::from_secs(2));
                let state: State<Sidecar> = monitor_handle.state();
                if state.shutting_down.load(Ordering::SeqCst) {
                    break;
                }
                let exited = {
                    let mut guard = state.child.lock().unwrap();
                    match guard.as_mut() {
                        Some(child) => matches!(child.try_wait(), Ok(Some(_))),
                        None => false,
                    }
                };
                if exited && !state.shutting_down.load(Ordering::SeqCst) {
                    respawn_and_notify(&monitor_handle, "beklenmedik çöküşten sonra yeniden başlatıldı");
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("tauri başlatılamadı")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                // Bayrağı önce koy: izleyici thread'i bu kasıtlı ölümü "çökme" sanıp
                // yeniden doğurmaya kalkışmasın.
                let state: State<Sidecar> = app.state();
                state.shutting_down.store(true, Ordering::SeqCst);
                let child = state.child.lock().unwrap().take();
                if let Some(mut child) = child {
                    kill_tree(&mut child);
                }
            }
        });
}
