import {
  addEdge,
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  useEdgesState,
  useNodesState,
  type Connection,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createWorkflow,
  deleteWorkflow,
  fetchNodeTypes,
  fetchWorkflow,
  fetchWorkflowRun,
  fetchWorkflows,
  runWorkflow,
  updateWorkflow,
  type NodeTypeInfo,
  type WorkflowGraphData,
  type WorkflowInfo,
  type WorkflowRunInfo,
} from "../api";
import { useStudio } from "../state";

type NodeStatus = "idle" | "running" | "done" | "failed" | "skipped";

interface StudioNodeData extends Record<string, unknown> {
  typeId: string;
  nodeType: NodeTypeInfo | undefined;
  params: Record<string, unknown>;
  status: NodeStatus;
  fromCache: boolean;
  resultError: string;
  onParamChange: (nodeId: string, paramName: string, value: unknown) => void;
}

type StudioNode = Node<StudioNodeData, "studioNode">;

function portTypesCompatible(a: string, b: string): boolean {
  return a === b || a === "any" || b === "any";
}

const STATUS_COLOR: Record<NodeStatus, string> = {
  idle: "#444",
  running: "#d9a441",
  done: "#3fae5a",
  failed: "#c94f4f",
  skipped: "#666",
};

function newNodeId(typeId: string): string {
  const slug = typeId.replace(/\./g, "_");
  return `${slug}_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function newEdgeId(): string {
  return `e_${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;
}

function ParamField({
  nodeId,
  spec,
  value,
  onChange,
}: {
  nodeId: string;
  spec: NodeTypeInfo["params"][number];
  value: unknown;
  onChange: (nodeId: string, paramName: string, value: unknown) => void;
}) {
  const current = value ?? spec.default ?? "";
  if (spec.type === "boolean") {
    return (
      <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
        <input
          className="nodrag"
          type="checkbox"
          checked={Boolean(current)}
          onChange={(e) => onChange(nodeId, spec.name, e.target.checked)}
        />
        {spec.name}
      </label>
    );
  }
  if (spec.type === "enum" && spec.choices) {
    return (
      <label style={{ display: "flex", flexDirection: "column", fontSize: 11, gap: 2 }}>
        {spec.name}
        <select
          className="nodrag"
          value={String(current)}
          onChange={(e) => onChange(nodeId, spec.name, e.target.value)}
        >
          {spec.choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </label>
    );
  }
  if (spec.type === "number") {
    return (
      <label style={{ display: "flex", flexDirection: "column", fontSize: 11, gap: 2 }}>
        {spec.name}
        <input
          className="nodrag"
          type="number"
          value={String(current)}
          onChange={(e) => onChange(nodeId, spec.name, e.target.value === "" ? "" : Number(e.target.value))}
        />
      </label>
    );
  }
  if (spec.type === "json") {
    return (
      <label style={{ display: "flex", flexDirection: "column", fontSize: 11, gap: 2 }}>
        {spec.name} (json)
        <textarea
          className="nodrag"
          rows={2}
          value={typeof current === "string" ? current : JSON.stringify(current)}
          onChange={(e) => {
            try {
              onChange(nodeId, spec.name, JSON.parse(e.target.value));
            } catch {
              onChange(nodeId, spec.name, e.target.value);
            }
          }}
        />
      </label>
    );
  }
  return (
    <label style={{ display: "flex", flexDirection: "column", fontSize: 11, gap: 2 }}>
      {spec.name}
      <input
        className="nodrag"
        type="text"
        value={String(current)}
        onChange={(e) => onChange(nodeId, spec.name, e.target.value)}
      />
    </label>
  );
}

function StudioNodeView({ id, data }: NodeProps<StudioNode>) {
  const nt = data.nodeType;
  const color = STATUS_COLOR[data.status];
  if (!nt) {
    return (
      <div style={{ padding: 8, border: "1px solid #c94f4f", borderRadius: 8, background: "#241414" }}>
        bilinmeyen düğüm türü: {data.typeId}
      </div>
    );
  }
  return (
    <div
      style={{
        minWidth: 200,
        border: `2px solid ${color}`,
        borderRadius: 8,
        background: "#1c1c1c",
        color: "#ddd",
        fontSize: 12,
      }}
    >
      <div
        style={{
          padding: "6px 10px",
          borderBottom: "1px solid #333",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontWeight: 600,
        }}
      >
        <span>{nt.label}</span>
        {data.fromCache && <span style={{ fontSize: 10, color: "#7ab8e0" }}>cache</span>}
      </div>
      {nt.inputs.map((p, i) => (
        <Handle
          key={`in-${p.name}`}
          id={p.name}
          type="target"
          position={Position.Left}
          style={{ top: 34 + i * 18, background: "#7ab8e0" }}
          title={`${p.name}: ${p.type}${p.required ? " (zorunlu)" : ""}`}
        />
      ))}
      {nt.outputs.map((p, i) => (
        <Handle
          key={`out-${p.name}`}
          id={p.name}
          type="source"
          position={Position.Right}
          style={{ top: 34 + i * 18, background: "#7ab8e0" }}
          title={`${p.name}: ${p.type}`}
        />
      ))}
      {nt.params.length > 0 && (
        <div style={{ padding: 8, display: "flex", flexDirection: "column", gap: 6 }}>
          {nt.params.map((spec) => (
            <ParamField
              key={spec.name}
              nodeId={id}
              spec={spec}
              value={data.params[spec.name]}
              onChange={data.onParamChange}
            />
          ))}
        </div>
      )}
      {data.resultError && (
        <div style={{ padding: "4px 8px", color: "#e08080", fontSize: 11, borderTop: "1px solid #333" }}>
          {data.resultError}
        </div>
      )}
    </div>
  );
}

const nodeTypesMap = { studioNode: StudioNodeView };

function graphToFlow(
  graph: WorkflowGraphData,
  nodeTypeById: Map<string, NodeTypeInfo>,
  onParamChange: (nodeId: string, paramName: string, value: unknown) => void,
): { nodes: StudioNode[]; edges: Edge[] } {
  const nodes: StudioNode[] = graph.nodes.map((n, i) => {
    const ui = (n.ui ?? {}) as { x?: number; y?: number };
    return {
      id: n.id,
      type: "studioNode",
      position: { x: ui.x ?? 80 + (i % 4) * 240, y: ui.y ?? 60 + Math.floor(i / 4) * 200 },
      data: {
        typeId: n.type,
        nodeType: nodeTypeById.get(n.type),
        params: { ...n.params },
        status: "idle",
        fromCache: false,
        resultError: "",
        onParamChange,
      },
    };
  });
  const edges: Edge[] = graph.edges.map((e) => ({
    id: e.id,
    source: e.from_node,
    sourceHandle: e.from_port,
    target: e.to_node,
    targetHandle: e.to_port,
  }));
  return { nodes, edges };
}

function flowToGraph(nodes: StudioNode[], edges: Edge[]): WorkflowGraphData {
  return {
    nodes: nodes.map((n) => ({
      id: n.id,
      type: n.data.typeId,
      params: n.data.params,
      ui: { x: n.position.x, y: n.position.y },
    })),
    edges: edges.map((e) => ({
      id: e.id,
      from_node: e.source,
      from_port: e.sourceHandle ?? "",
      to_node: e.target,
      to_port: e.targetHandle ?? "",
    })),
  };
}

export default function WorkflowPanel() {
  const { events } = useStudio();
  const [nodeTypes, setNodeTypes] = useState<NodeTypeInfo[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowInfo[]>([]);
  const [currentId, setCurrentId] = useState<string>("");
  const [name, setName] = useState("Adsız workflow");
  const [seed, setSeed] = useState(0);
  const [nodes, setNodes, onNodesChange] = useNodesState<StudioNode>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [saveError, setSaveError] = useState("");
  const [connectError, setConnectError] = useState("");
  const [run, setRun] = useState<WorkflowRunInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const lastEvent = useRef(0);
  const pollTimer = useRef<number | null>(null);

  const nodeTypeById = useMemo(() => new Map(nodeTypes.map((t) => [t.type_id, t])), [nodeTypes]);

  const onParamChange = useCallback(
    (nodeId: string, paramName: string, value: unknown) => {
      setNodes((nds) =>
        nds.map((n) =>
          n.id === nodeId ? { ...n, data: { ...n.data, params: { ...n.data.params, [paramName]: value } } } : n,
        ),
      );
    },
    [setNodes],
  );

  const refreshWorkflows = useCallback(() => {
    fetchWorkflows().then(setWorkflows).catch(() => undefined);
  }, []);

  useEffect(() => {
    fetchNodeTypes().then(setNodeTypes).catch((e) => setSaveError(String(e)));
    refreshWorkflows();
  }, [refreshWorkflows]);

  // Node türleri asenkron geldiği için, onlardan önce yüklenmiş bir grafiğin
  // düğümleri "bilinmeyen tür" görünür — tür defteri gelince yeniden bağla.
  useEffect(() => {
    if (nodeTypeById.size === 0) return;
    setNodes((nds) =>
      nds.map((n) =>
        n.data.nodeType ? n : { ...n, data: { ...n.data, nodeType: nodeTypeById.get(n.data.typeId) } },
      ),
    );
  }, [nodeTypeById, setNodes]);

  const applyRun = useCallback((r: WorkflowRunInfo) => {
    setRun(r);
    setNodes((nds) =>
      nds.map((n) => {
        const res = r.node_results[n.id];
        if (!res) return n;
        return {
          ...n,
          data: {
            ...n.data,
            status: res.status,
            fromCache: res.from_cache,
            resultError: res.error,
          },
        };
      }),
    );
  }, [setNodes]);

  const pollRun = useCallback(
    (runId: string) => {
      if (pollTimer.current) window.clearInterval(pollTimer.current);
      pollTimer.current = window.setInterval(() => {
        fetchWorkflowRun(runId)
          .then((r) => {
            applyRun(r);
            if (r.status !== "running" && pollTimer.current) {
              window.clearInterval(pollTimer.current);
              pollTimer.current = null;
            }
          })
          .catch(() => undefined);
      }, 1000);
    },
    [applyRun],
  );

  useEffect(() => () => {
    if (pollTimer.current) window.clearInterval(pollTimer.current);
  }, []);

  // Workflow'un kendi olayları geldiğinde (bkz. studio_core.events workflow.*)
  // bir sonraki poll'u beklemeden anında yenile — MeshPanel'deki job.* deseniyle aynı.
  useEffect(() => {
    let hit = false;
    for (let i = lastEvent.current; i < events.length; i++) {
      const e = events[i];
      if (e.topic.startsWith("workflow.") && run && (e.data as { run_id?: string }).run_id === run.id) {
        hit = true;
      }
    }
    lastEvent.current = events.length;
    if (hit) fetchWorkflowRun(run!.id).then(applyRun).catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [events]);

  const loadWorkflow = async (id: string) => {
    if (!id) {
      setCurrentId("");
      setName("Adsız workflow");
      setNodes([]);
      setEdges([]);
      setRun(null);
      return;
    }
    const wf = await fetchWorkflow(id);
    setCurrentId(wf.id);
    setName(wf.name);
    const { nodes: n, edges: e } = graphToFlow(wf.graph, nodeTypeById, onParamChange);
    setNodes(n);
    setEdges(e);
    setRun(null);
    setSaveError("");
  };

  const addPaletteNode = (nt: NodeTypeInfo) => {
    const params: Record<string, unknown> = {};
    for (const p of nt.params) if (p.default !== null && p.default !== undefined) params[p.name] = p.default;
    const id = newNodeId(nt.type_id);
    const idx = nodes.length;
    const node: StudioNode = {
      id,
      type: "studioNode",
      position: { x: 80 + (idx % 4) * 240, y: 60 + Math.floor(idx / 4) * 200 },
      data: {
        typeId: nt.type_id,
        nodeType: nt,
        params,
        status: "idle",
        fromCache: false,
        resultError: "",
        onParamChange,
      },
    };
    setNodes((nds) => [...nds, node]);
  };

  const portsOf = useCallback(
    (conn: Connection | Edge) => {
      const src = nodes.find((n) => n.id === conn.source);
      const tgt = nodes.find((n) => n.id === conn.target);
      return {
        srcPort: src?.data.nodeType?.outputs.find((p) => p.name === conn.sourceHandle),
        tgtPort: tgt?.data.nodeType?.inputs.find((p) => p.name === conn.targetHandle),
      };
    },
    [nodes],
  );

  // Tip-uyumsuz bağlantı sürüklenirken engellenir (plan.md Faz 10 editör maddesi):
  // React Flow bunu kullanarak geçersiz hedefi bırakılamaz yapar.
  const isValidConnection = useCallback(
    (conn: Connection | Edge) => {
      const { srcPort, tgtPort } = portsOf(conn);
      return Boolean(srcPort && tgtPort && portTypesCompatible(srcPort.type, tgtPort.type));
    },
    [portsOf],
  );

  const onConnect = useCallback(
    (conn: Connection) => {
      const { srcPort, tgtPort } = portsOf(conn);
      if (!srcPort || !tgtPort) return;
      if (!portTypesCompatible(srcPort.type, tgtPort.type)) {
        setConnectError(`tip uyumsuz: ${srcPort.type} → ${tgtPort.type}`);
        window.setTimeout(() => setConnectError(""), 3000);
        return;
      }
      // Bir giriş portu yalnızca tek bir kaynaktan beslenebilir — yenisi eskisini
      // değiştirir (executor zaten port başına tek değer okur).
      setEdges((eds) =>
        addEdge(
          { ...conn, id: newEdgeId() },
          eds.filter((e) => !(e.target === conn.target && e.targetHandle === conn.targetHandle)),
        ),
      );
    },
    [portsOf, setEdges],
  );

  const doSave = async (): Promise<string | null> => {
    setSaveError("");
    const graph = flowToGraph(nodes, edges);
    try {
      const wf = currentId
        ? await updateWorkflow(currentId, name, graph)
        : await createWorkflow(name, graph);
      setCurrentId(wf.id);
      refreshWorkflows();
      return wf.id;
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
      return null;
    }
  };

  const doRun = async () => {
    setBusy(true);
    try {
      const id = await doSave();
      if (!id) return;
      const r = await runWorkflow(id, seed);
      applyRun(r);
      pollRun(r.id);
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const doDelete = async () => {
    if (!currentId) return;
    await deleteWorkflow(currentId);
    refreshWorkflows();
    loadWorkflow("");
  };

  const categories = useMemo(() => {
    const byCat = new Map<string, NodeTypeInfo[]>();
    for (const nt of nodeTypes) {
      const list = byCat.get(nt.category) ?? [];
      list.push(nt);
      byCat.set(nt.category, list);
    }
    return [...byCat.entries()];
  }, [nodeTypes]);

  return (
    <div style={{ display: "flex", height: "100%", fontSize: 13 }}>
      <div
        style={{
          width: 220,
          padding: 10,
          borderRight: "1px solid #333",
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div>
          <label style={{ display: "block", marginBottom: 4 }}>Workflow</label>
          <select
            style={{ width: "100%" }}
            value={currentId}
            onChange={(e) => loadWorkflow(e.target.value)}
          >
            <option value="">— yeni —</option>
            {workflows.map((w) => (
              <option key={w.id} value={w.id}>
                {w.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label style={{ display: "block", marginBottom: 4 }}>Düğüm paleti</label>
          {categories.map(([cat, list]) => (
            <div key={cat} style={{ marginBottom: 8 }}>
              <div style={{ color: "var(--muted)", fontSize: 11, margin: "4px 0" }}>{cat}</div>
              {list.map((nt) => (
                <button
                  key={nt.type_id}
                  onClick={() => addPaletteNode(nt)}
                  style={{ display: "block", width: "100%", textAlign: "left", marginBottom: 4 }}
                >
                  {nt.label}
                </button>
              ))}
            </div>
          ))}
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        <div className="toolbar" style={{ padding: 8, borderBottom: "1px solid #333", margin: 0 }}>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            style={{ minWidth: 160 }}
          />
          <span className="gen-num">
            seed
            <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
          </span>
          <button onClick={doSave}>Kaydet</button>
          <button onClick={doRun} disabled={busy || run?.status === "running"}>
            Çalıştır
          </button>
          {currentId && <button onClick={doDelete}>Sil</button>}
          {run && <span className={`badge ${run.status === "failed" ? "err" : run.status === "done" ? "ok" : ""}`}>{run.status}</span>}
        </div>
        {(saveError || connectError || run?.error) && (
          <div style={{ padding: "6px 10px", color: "#e08080", fontSize: 12, borderBottom: "1px solid #333" }}>
            {saveError || connectError || run?.error}
          </div>
        )}
        <div style={{ flex: 1, minHeight: 0 }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            isValidConnection={isValidConnection}
            nodeTypes={nodeTypesMap}
            colorMode="dark"
            fitView
          >
            <Background />
            <Controls />
            <MiniMap />
          </ReactFlow>
        </div>
      </div>
    </div>
  );
}
