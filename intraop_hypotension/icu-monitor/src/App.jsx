import React, { useState } from 'react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://localhost:8000/predict", { method: "POST", body: formData });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      alert("System Offline: Check Backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ backgroundColor: '#0f172a', minHeight: '100vh', color: '#f8fafc', fontFamily: 'sans-serif', padding: '20px' }}>
      {}
      <nav style={{ borderBottom: '1px solid #1e293b', paddingBottom: '20px', marginBottom: '40px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ color: '#38bdf8', margin: 0 }}>VITAL-AI <span style={{ color: '#94a3b8', fontSize: '0.8rem' }}>v2.4</span></h2>
        <div style={{ fontSize: '14px', color: '#94a3b8' }}>System Status: <span style={{ color: '#4ade80' }}>● Online</span></div>
      </nav>

      <div style={{ maxWidth: '1100px', margin: 'auto', display: 'grid', gridTemplateColumns: '350px 1fr', gap: '30px' }}>
        
        {}
        <section>
          <div style={{ backgroundColor: '#1e293b', padding: '25px', borderRadius: '16px', border: '1px solid #334155' }}>
            <h4 style={{ marginTop: 0, color: '#f1f5f9' }}>Data Input</h4>
            <div style={{ margin: '20px 0', padding: '20px', border: '2px dashed #334155', borderRadius: '12px', textAlign: 'center' }}>
              <input type="file" accept=".csv" onChange={(e) => setSelectedFile(e.target.files[0])} style={{ width: '100%', cursor: 'pointer' }} />
              {selectedFile && <p style={{ fontSize: '12px', color: '#38bdf8', marginTop: '10px' }}>Selected: {selectedFile.name}</p>}
            </div>
            <button 
              onClick={handleUpload} 
              disabled={loading || !selectedFile}
              style={{ width: '100%', padding: '14px', borderRadius: '8px', border: 'none', backgroundColor: '#38bdf8', color: '#0f172a', fontWeight: 'bold', cursor: 'pointer', opacity: loading ? 0.6 : 1 }}
            >
              {loading ? "CALCULATING..." : "ANALYZE VITALS"}
            </button>
          </div>
        </section>

        {}
        {}
<main style={{ maxWidth: '800px' }}> {}
  {result && (
    <div style={{ animation: 'slideUp 0.4s ease-out' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px', marginBottom: '20px' }}>
        <StatCard label="PATIENT ID" value={result.patient_id} color="#f1f5f9" />
        <StatCard label="DETECTION TIME" value={result.occurrence_time} color="#38bdf8" />
        <StatCard label="RISK SCORE" value={`${(result.risk_score * 100).toFixed(1)}%`} color={result.risk_score > 0.5 ? '#f87171' : '#4ade80'} />
      </div>

      <div style={{ backgroundColor: result.risk_score > 0.5 ? 'rgba(248, 113, 113, 0.1)' : 'rgba(74, 222, 128, 0.1)', padding: '40px', borderRadius: '16px', border: `1px solid ${result.risk_score > 0.5 ? '#f87171' : '#4ade80'}`, textAlign: 'center', marginBottom: '20px' }}>
        <span style={{ fontSize: '0.8rem', fontWeight: 'bold', letterSpacing: '2px', color: result.risk_score > 0.5 ? '#f87171' : '#4ade80' }}>FINAL DIAGNOSIS</span>
        <h1 style={{ fontSize: '3rem', margin: '10px 0', color: result.risk_score > 0.5 ? '#f87171' : '#4ade80' }}>{result.diagnosis}</h1>
        <p style={{ color: '#94a3b8' }}>
          {result.risk_score > 0.5 
            ? `Critical event predicted at ${result.occurrence_time}.` 
            : "No immediate intervention required."}
        </p>
      </div>
      
      {}
      <div style={{ fontSize: '12px', color: '#475569', textAlign: 'right' }}>
        System processed: {result.filename_processed}
      </div>
    </div>
  )}
</main>
      </div>

      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}

function StatCard({ label, value, color }) {
  return (
    <div style={{ backgroundColor: '#1e293b', padding: '20px', borderRadius: '12px', border: '1px solid #334155' }}>
      <div style={{ fontSize: '11px', color: '#64748b', fontWeight: 'bold', marginBottom: '8px' }}>{label}</div>
      <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: color }}>{value}</div>
    </div>
  );
}

export default App;