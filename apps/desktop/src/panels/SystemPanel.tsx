import { useStudio } from "../state";

export default function SystemPanel() {
  const { info } = useStudio();
  return (
    <div className="panel-body">
      {!info ? (
        <p>Backend bekleniyor… (uv run studio-api)</p>
      ) : (
        <dl className="kv">
          <dt>OS</dt>
          <dd>
            {info.os} {info.arch}
          </dd>
          <dt>Python</dt>
          <dd>{info.python}</dd>
          <dt>CPU</dt>
          <dd>{info.cpu_count} çekirdek</dd>
          <dt>RAM</dt>
          <dd>
            {(info.ram_available_mb / 1024).toFixed(1)} GB boş /{" "}
            {(info.ram_total_mb / 1024).toFixed(1)} GB
          </dd>
          <dt>Disk (veri dizini)</dt>
          <dd>{info.disk_free_gb} GB boş</dd>
          <dt>GPU backend</dt>
          <dd>{info.gpu_backend.toUpperCase()}</dd>
          {info.gpus.map((g) => (
            <GpuRow key={g.index} name={g.name} used={g.vram_used_mb} total={g.vram_total_mb} />
          ))}
          <dt>Veri dizini</dt>
          <dd>{info.data_dir}</dd>
        </dl>
      )}
    </div>
  );
}

function GpuRow({ name, used, total }: { name: string; used: number; total: number }) {
  return (
    <>
      <dt>GPU</dt>
      <dd>
        {name} — VRAM {(used / 1024).toFixed(1)} / {(total / 1024).toFixed(1)} GB
      </dd>
    </>
  );
}
