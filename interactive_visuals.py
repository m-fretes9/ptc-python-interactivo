from __future__ import annotations

import html
import json
from typing import Any, Mapping


def _js(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def ptc_optical_component_html(
    config: Mapping[str, Any],
    snapshot: Mapping[str, float],
    selected_node: int,
) -> str:
    """Return a self-contained HTML/CSS/JS PTC optical visualizer.

    The widget is intentionally dependency-free so it can be embedded with
    ``streamlit.components.v1.html`` and deployed on Streamlit Cloud/GitHub.

    The heat map is an *optical incidence distribution* on the absorber outer
    circumference. It distributes the model's already-computed absorbed solar
    power of the selected axial volume according to the density of ideal
    reflected-ray hits. It is not a 2-D conduction/CFD temperature field.
    """
    g = config["geometry"]
    model = config["model"]
    geometry = {
        "W": float(g["W"]),
        "L": float(g["L"]),
        "f": float(g["f"]),
        "D3": float(g["D3"]),
        "D5": float(g["D5"]),
        "Nseg": int(g["Nseg"]),
        "hasGlass": bool(model.get("has_glass", True)),
    }
    data = {
        "node": int(selected_node) + 1,
        "LAT": float(snapshot.get("LAT_h", 0.0)),
        "theta": float(snapshot.get("theta_deg", 0.0)),
        "DNI": float(snapshot.get("DNI_W_m2", 0.0)),
        "Qsolar": float(snapshot.get("Qsolar_abs_node_W", 0.0)),
        "Tabs": float(snapshot.get("Tabs_C", 0.0)),
        "Tglass": float(snapshot.get("Tglass_C", 0.0)),
    }

    # Avoid accidental HTML injection if future labels become user-defined.
    title = html.escape(f"PTC · nodo axial {data['node']}")

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --ink:#111827; --muted:#64748b; --line:#cbd5e1; --panel:#ffffff;
    --soft:#f8fafc; --sun:#f59e0b; --ray:#2563eb; --hot:#ef4444;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;padding:0;background:transparent;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;color:var(--ink)}}
  .card{{background:#fff;border:1px solid #e2e8f0;border-radius:18px;overflow:hidden;box-shadow:0 8px 24px rgba(15,23,42,.05)}}
  .top{{padding:14px 16px 12px;border-bottom:1px solid #eef2f7;background:linear-gradient(180deg,#fff,#fbfdff)}}
  .title{{font-weight:720;font-size:16px;letter-spacing:-.01em}}
  .sub{{font-size:12px;color:var(--muted);margin-top:3px}}
  .toolbar{{display:grid;grid-template-columns:auto minmax(180px,1fr) minmax(180px,1fr) auto;gap:12px;align-items:end;padding:12px 16px;background:#fbfdff;border-bottom:1px solid #eef2f7}}
  .seg{{display:flex;padding:3px;background:#eef2f7;border-radius:11px;gap:2px}}
  .seg button{{border:0;background:transparent;padding:8px 10px;border-radius:8px;cursor:pointer;font-weight:650;font-size:12px;color:#475569;white-space:nowrap}}
  .seg button.active{{background:#fff;color:#0f172a;box-shadow:0 1px 4px rgba(15,23,42,.12)}}
  .ctrl label{{display:flex;justify-content:space-between;gap:8px;font-size:11px;color:#475569;margin-bottom:6px}}
  .ctrl b{{color:#0f172a}}
  input[type=range]{{width:100%;accent-color:#111827;cursor:pointer}}
  .btn{{border:1px solid #dbe3ec;background:#fff;border-radius:10px;padding:8px 11px;cursor:pointer;font-weight:650;color:#334155}}
  .btn:hover{{background:#f8fafc}}
  .scene{{position:relative;background:#fff;padding:8px 10px 4px}}
  svg{{width:100%;height:510px;display:block}}
  .mirror{{fill:none;stroke:#334155;stroke-width:4;stroke-linecap:round}}
  .glass{{fill:none;stroke:#94a3b8;stroke-width:3;stroke-dasharray:7 6}}
  .absorber-base{{fill:none;stroke:#111827;stroke-width:13}}
  .incoming{{stroke:var(--sun);stroke-width:2.4;opacity:.92}}
  .reflected{{stroke:var(--ray);stroke-width:3.0;opacity:.95}}
  .soft-ray{{stroke:#2563eb;stroke-width:1;opacity:.08}}
  .hit{{fill:#111827;stroke:#fff;stroke-width:1.5}}
  .focus{{fill:#0f172a}}
  .travel{{fill:#fff;stroke:#111827;stroke-width:2;filter:drop-shadow(0 1px 2px rgba(15,23,42,.25))}}
  .legend{{position:absolute;right:22px;top:24px;width:150px;background:rgba(255,255,255,.92);backdrop-filter:blur(6px);border:1px solid #e2e8f0;border-radius:12px;padding:10px 11px;font-size:11px;color:#475569}}
  .gradient{{height:8px;border-radius:999px;margin:7px 0 4px;background:linear-gradient(90deg,#1d4ed8,#06b6d4,#fde047,#f97316,#dc2626)}}
  .metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;padding:10px 16px 14px;background:#fff}}
  .metric{{border:1px solid #e8edf3;border-radius:12px;padding:9px 10px;background:#fff}}
  .metric .k{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.04em}}
  .metric .v{{font-size:15px;font-weight:720;margin-top:2px}}
  .note{{padding:0 16px 14px;font-size:11px;color:#64748b;line-height:1.35}}
  .tiny{{font-size:10px;fill:#64748b}}
  @media(max-width:850px){{.toolbar{{grid-template-columns:1fr 1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}svg{{height:450px}}}}
</style>
</head>
<body>
<div class="card">
  <div class="top"><div class="title">{title} · óptica interactiva</div><div class="sub">Incidencia, reflexión y distribución óptica sobre el absorbedor</div></div>
  <div class="toolbar">
    <div class="seg">
      <button id="rayBtn" class="active">Seguidor de rayo</button>
      <button id="heatBtn">Mapa de calor</button>
    </div>
    <div class="ctrl"><label><span>Ángulo de incidencia</span><b id="angleOut">0.0°</b></label><input id="angle" type="range" min="-25" max="25" value="0" step="0.25"></div>
    <div class="ctrl"><label><span>Posición del rayo en la abertura</span><b id="rayOut">50%</b></label><input id="rayPos" type="range" min="4" max="96" value="50" step="0.5"></div>
    <button class="btn" id="playBtn">▶ Animar rayo</button>
  </div>
  <div class="scene">
    <svg id="svg" viewBox="0 0 1000 620" preserveAspectRatio="xMidYMid meet">
      <defs><clipPath id="sceneClip"><rect x="0" y="0" width="1000" height="620" rx="12"/></clipPath></defs>
      <g id="grid" clip-path="url(#sceneClip)"></g>
      <path id="mirror" class="mirror"></path>
      <circle id="glass" class="glass"></circle>
      <g id="heatArcs"></g>
      <circle id="absorber" class="absorber-base"></circle>
      <circle id="focus" r="4.5" class="focus"></circle>
      <g id="rayCloud"></g>
      <g id="singleRay"></g>
      <circle id="travel" r="7" class="travel" style="display:none"></circle>
      <text id="thetaText" x="22" y="28" class="tiny"></text>
    </svg>
    <div class="legend" id="legend"><b>Seguidor de rayo</b><div style="margin-top:5px">Amarillo: rayo incidente<br>Azul: rayo reflejado</div></div>
  </div>
  <div class="metrics">
    <div class="metric"><div class="k">DNI</div><div class="v" id="dni">—</div></div>
    <div class="metric"><div class="k">Q solar / nodo</div><div class="v" id="qsolar">—</div></div>
    <div class="metric"><div class="k">T absorbedor</div><div class="v" id="tabs">—</div></div>
    <div class="metric"><div class="k">Rayos que impactan</div><div class="v" id="hits">—</div></div>
    <div class="metric"><div class="k">Pico óptico estimado</div><div class="v" id="peak">—</div></div>
  </div>
  <div class="note">Distribución óptica relativa de la potencia absorbida sobre la circunferencia externa del receptor.</div>
</div>
<script>
const G={_js(geometry)}, D={_js(data)};
const svg=document.getElementById('svg'), NS='http://www.w3.org/2000/svg';
const angle=document.getElementById('angle'), rayPos=document.getElementById('rayPos');
angle.value=Math.max(-25,Math.min(25,D.theta||0));
let mode='ray', animHandle=null, animStart=null;

// Physical coordinates: y grows upward. SVG coordinates: y grows downward.
const xMin=-G.W*0.62, xMax=G.W*0.62;
const yEdge=(G.W/2)**2/(4*G.f), yMax=Math.max(yEdge+0.18*G.W,G.f+0.32*G.W), yMin=-0.15*G.W;
function sx(x){{return 70+(x-xMin)/(xMax-xMin)*860;}}
function sy(y){{return 570-(y-yMin)/(yMax-yMin)*500;}}
function P(x,y){{return [sx(x),sy(y)];}}
function norm(v){{const n=Math.hypot(v[0],v[1])||1;return [v[0]/n,v[1]/n];}}
function dot(a,b){{return a[0]*b[0]+a[1]*b[1];}}
function reflect(d,n){{d=norm(d);n=norm(n);const k=2*dot(d,n);return norm([d[0]-k*n[0],d[1]-k*n[1]]);}}
function parabolaY(x){{return x*x/(4*G.f);}}
function intersectParabola(x0,y0,d){{
  const a=d[0]*d[0], b=2*x0*d[0]-4*G.f*d[1], c=x0*x0-4*G.f*y0;
  let roots=[];
  if(Math.abs(a)<1e-12){{ if(Math.abs(b)>1e-12) roots=[-c/b]; }} else {{
    const disc=b*b-4*a*c;if(disc>=0)roots=[(-b-Math.sqrt(disc))/(2*a),(-b+Math.sqrt(disc))/(2*a)];
  }}
  roots=roots.filter(t=>Number.isFinite(t)&&t>=0).sort((a,b)=>a-b); if(!roots.length)return null;
  const t=roots[0];return [x0+t*d[0],y0+t*d[1]];
}}
function intersectCircle(o,d,c,r){{
  d=norm(d);const px=o[0]-c[0],py=o[1]-c[1];const b=2*(px*d[0]+py*d[1]);const cc=px*px+py*py-r*r;const disc=b*b-4*cc;
  if(disc<0)return null;const roots=[(-b-Math.sqrt(disc))/2,(-b+Math.sqrt(disc))/2].filter(t=>t>1e-6).sort((a,b)=>a-b);if(!roots.length)return null;
  return [o[0]+roots[0]*d[0],o[1]+roots[0]*d[1]];
}}
function trace(frac,deg){{
  const th=deg*Math.PI/180, d=norm([Math.sin(th),-Math.cos(th)]), x0=(frac-.5)*G.W, y0=yMax;
  const m=intersectParabola(x0,y0,d);if(!m||Math.abs(m[0])>G.W/2+1e-6)return null;
  const n=norm([-m[0]/(2*G.f),1]), r=reflect(d,n), c=[0,G.f], hit=intersectCircle([m[0]+1e-7*r[0],m[1]+1e-7*r[1]],r,c,G.D3/2);
  return {{launch:[x0,y0],mirror:m,dir:d,refl:r,hit}};
}}
function el(tag,attrs={{}}){{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e;}}
function clear(id){{document.getElementById(id).innerHTML='';}}
function polyline(parent,pts,cls){{const p=el('polyline',{{points:pts.map(q=>P(q[0],q[1]).join(',')).join(' '),fill:'none',class:cls}});parent.appendChild(p);return p;}}

function drawBase(){{
  const xs=Array.from({{length:181}},(_,i)=>-G.W/2+i*G.W/180), pts=xs.map(x=>P(x,parabolaY(x)));
  document.getElementById('mirror').setAttribute('d','M '+pts.map(p=>p.join(',')).join(' L '));
  const c=P(0,G.f), ra=(G.D3/2)*860/(xMax-xMin), rg=(G.D5/2)*860/(xMax-xMin);
  for(const id of ['absorber','glass']){{document.getElementById(id).setAttribute('cx',c[0]);document.getElementById(id).setAttribute('cy',c[1]);}}
  document.getElementById('absorber').setAttribute('r',ra);document.getElementById('glass').setAttribute('r',rg);
  document.getElementById('glass').style.display=G.hasGlass?'block':'none';
  document.getElementById('focus').setAttribute('cx',c[0]);document.getElementById('focus').setAttribute('cy',c[1]);
  const grid=document.getElementById('grid');grid.innerHTML='';
  for(let i=0;i<8;i++){{const y=70+i*62;grid.appendChild(el('line',{{x1:55,y1:y,x2:945,y2:y,stroke:'#f1f5f9','stroke-width':1}}));}}
}}
function colorScale(t){{t=Math.max(0,Math.min(1,t));const stops=[[29,78,216],[6,182,212],[253,224,71],[249,115,22],[220,38,38]];const z=t*(stops.length-1),i=Math.min(stops.length-2,Math.floor(z)),f=z-i,a=stops[i],b=stops[i+1];return `rgb(${{Math.round(a[0]+f*(b[0]-a[0]))}},${{Math.round(a[1]+f*(b[1]-a[1]))}},${{Math.round(a[2]+f*(b[2]-a[2]))}})`;}}
function arcPath(cx,cy,r,a0,a1){{const p0=[cx+r*Math.cos(a0),cy-r*Math.sin(a0)],p1=[cx+r*Math.cos(a1),cy-r*Math.sin(a1)];return `M ${{p0[0]}} ${{p0[1]}} A ${{r}} ${{r}} 0 0 0 ${{p1[0]}} ${{p1[1]}}`;}}

function renderSingle(){{
  clear('singleRay');clear('rayCloud');clear('heatArcs');
  const frac=parseFloat(rayPos.value)/100, deg=parseFloat(angle.value), t=trace(frac,deg), g=document.getElementById('singleRay');
  if(t){{polyline(g,[t.launch,t.mirror],'incoming');const ext=t.hit||[t.mirror[0]+t.refl[0]*G.W,t.mirror[1]+t.refl[1]*G.W];polyline(g,[t.mirror,ext],'reflected');
    const pm=P(t.mirror[0],t.mirror[1]);g.appendChild(el('circle',{{cx:pm[0],cy:pm[1],r:5,fill:'#7c3aed'}}));
    if(t.hit){{const ph=P(t.hit[0],t.hit[1]);g.appendChild(el('circle',{{cx:ph[0],cy:ph[1],r:6,class:'hit'}}));document.getElementById('hits').textContent='1 / 1';}}else document.getElementById('hits').textContent='0 / 1';
  }} else document.getElementById('hits').textContent='0 / 1';
  document.getElementById('peak').textContent='—';document.getElementById('legend').innerHTML='<b>Seguidor de rayo</b><div style="margin-top:5px">Amarillo: incidente<br>Azul: reflejado</div>';
}}
function renderHeat(){{
  clear('singleRay');clear('rayCloud');clear('heatArcs');
  const deg=parseFloat(angle.value), bins=72, counts=Array(bins).fill(0), cloud=document.getElementById('rayCloud');let hits=0;
  for(let i=0;i<180;i++){{const f=.03+i*.94/179,t=trace(f,deg);if(!t)continue;if(i%6===0){{const e=t.hit||[t.mirror[0]+t.refl[0]*G.W*.7,t.mirror[1]+t.refl[1]*G.W*.7];polyline(cloud,[t.launch,t.mirror,e],'soft-ray');}}
    if(t.hit){{hits++;let a=Math.atan2(t.hit[1]-G.f,t.hit[0]);if(a<0)a+=Math.PI*2;const b=Math.floor(a/(Math.PI*2)*bins)%bins;counts[b]++;}}
  }}
  const mx=Math.max(1,...counts), c=P(0,G.f), r=(G.D3/2)*860/(xMax-xMin), arcs=document.getElementById('heatArcs');
  for(let b=0;b<bins;b++){{const a0=b/bins*Math.PI*2,a1=(b+1)/bins*Math.PI*2,t=counts[b]/mx,p=el('path',{{d:arcPath(c[0],c[1],r,a0,a1),fill:'none',stroke:colorScale(t),'stroke-width':22,'stroke-linecap':'round',filter:'drop-shadow(0 0 3px rgba(249,115,22,.28))'}});p.appendChild(el('title'));p.firstChild.textContent=`Sector ${{b+1}} · intensidad relativa ${{(100*t).toFixed(1)}}%`;arcs.appendChild(p);}}
  document.getElementById('hits').textContent=`${{hits}} / 180`;
  const dx=G.L/G.Nseg, areaSector=(Math.PI*G.D3*dx)/bins, peakW=D.Qsolar*(mx/Math.max(1,hits))/areaSector;
  document.getElementById('peak').textContent=Number.isFinite(peakW)?`${{peakW.toFixed(0)}} W/m²`:'—';
  document.getElementById('legend').innerHTML='<b>Distribución óptica relativa</b><div class="gradient"></div><div style="display:flex;justify-content:space-between"><span>baja</span><span>alta</span></div>';
}}
function update(){{
  const deg=parseFloat(angle.value);document.getElementById('angleOut').textContent=deg.toFixed(2)+'°';document.getElementById('rayOut').textContent=parseFloat(rayPos.value).toFixed(1)+'%';document.getElementById('thetaText').textContent=`θ = ${{deg.toFixed(2)}}° · posición = ${{parseFloat(rayPos.value).toFixed(1)}}%`;
  document.getElementById('dni').textContent=`${{D.DNI.toFixed(1)}} W/m²`;document.getElementById('qsolar').textContent=`${{D.Qsolar.toFixed(2)}} W`;document.getElementById('tabs').textContent=`${{D.Tabs.toFixed(2)}} °C`;
  if(mode==='ray')renderSingle();else renderHeat();
}}
function setMode(m){{mode=m;document.getElementById('rayBtn').classList.toggle('active',m==='ray');document.getElementById('heatBtn').classList.toggle('active',m==='heat');document.getElementById('rayPos').disabled=(m==='heat');document.getElementById('playBtn').disabled=(m==='heat');document.getElementById('travel').style.display='none';if(animHandle)cancelAnimationFrame(animHandle);animHandle=null;update();}}
function animateRay(){{
  if(mode!=='ray')return;const t=trace(parseFloat(rayPos.value)/100,parseFloat(angle.value));if(!t)return;const pts=t.hit?[t.launch,t.mirror,t.hit]:[t.launch,t.mirror,[t.mirror[0]+t.refl[0]*G.W,t.mirror[1]+t.refl[1]*G.W]];const scr=pts.map(q=>P(q[0],q[1]));const lens=[Math.hypot(scr[1][0]-scr[0][0],scr[1][1]-scr[0][1]),Math.hypot(scr[2][0]-scr[1][0],scr[2][1]-scr[1][1])], total=lens[0]+lens[1];const dot=document.getElementById('travel');dot.style.display='block';animStart=null;if(animHandle)cancelAnimationFrame(animHandle);
  function step(ts){{if(animStart===null)animStart=ts;const u=((ts-animStart)%2200)/2200,dist=u*total;let p;if(dist<=lens[0]){{const f=dist/lens[0];p=[scr[0][0]+f*(scr[1][0]-scr[0][0]),scr[0][1]+f*(scr[1][1]-scr[0][1])];}}else{{const f=(dist-lens[0])/lens[1];p=[scr[1][0]+f*(scr[2][0]-scr[1][0]),scr[1][1]+f*(scr[2][1]-scr[1][1])];}}dot.setAttribute('cx',p[0]);dot.setAttribute('cy',p[1]);animHandle=requestAnimationFrame(step);}}
  animHandle=requestAnimationFrame(step);
}}
document.getElementById('rayBtn').onclick=()=>setMode('ray');document.getElementById('heatBtn').onclick=()=>setMode('heat');angle.oninput=update;rayPos.oninput=update;document.getElementById('playBtn').onclick=animateRay;
drawBase();update();
</script>
</body></html>"""


def thermal_circuit_component_html(
    config: Mapping[str, Any],
    snapshot: Mapping[str, float],
) -> str:
    """Circuito térmico interactivo en SVG/CSS/JS."""
    import math

    g = config["geometry"]
    model = config["model"]
    mats = config["materials"]
    env = config["environment"]
    nseg = max(int(g["Nseg"]), 1)
    dx = float(g["L"]) / nseg
    area_inner = math.pi * float(g["D2"]) * dx
    h_int = max(float(snapshot["h_internal_W_m2K"]), 1e-12)
    r12 = 1.0 / (h_int * area_inner)
    r23 = math.log(float(g["D3"]) / float(g["D2"])) / (
        2.0 * math.pi * float(mats["absorber"]["k"]) * dx
    )
    has_glass = bool(model.get("has_glass", True))
    tamb_c = float(env["Tamb_K"]) - 273.15
    tsky_c = float(snapshot.get("Tsky_C", tamb_c - float(env.get("sky_delta_K", 6.0))))

    qfluid = float(snapshot.get("Qfluid_W", 0.0))
    qsolar = float(snapshot.get("Qsolar_abs_node_W", 0.0))
    qradag = float(snapshot.get("Qrad_abs_glass_W", 0.0))
    qconvag = float(snapshot.get("Qconv_annulus_W", 0.0))
    qconvext = float(snapshot.get("Qconv_external_W", 0.0))
    qradsky = float(snapshot.get("Qrad_sky_W", 0.0))
    qsupports = float(snapshot.get("Qsupports_W", 0.0))
    qdh = -float(snapshot.get("Qadvection_W", 0.0))

    def finite_or_none(name: str) -> float | None:
        value = float(snapshot.get(name, float("nan")))
        return value if math.isfinite(value) else None

    data = {
        "hasGlass": has_glass,
        "node": int(snapshot.get("node_index", 0)) + 1,
        "LAT": float(snapshot.get("LAT_h", 0.0)),
        "T": {
            "Tagua": float(snapshot.get("Tf_C", 0.0)),
            "TabsI": float(snapshot.get("Tabs_C", 0.0)),
            "TabsE": float(snapshot.get("Tabs_C", 0.0)),
            "TvidI": float(snapshot.get("Tglass_C", 0.0)),
            "TvidE": float(snapshot.get("Tglass_C", 0.0)),
            "Tamb": tamb_c,
            "Tsky": tsky_c,
        },
        "R": {
            "r12": r12,
            "r23": r23,
            "r34rad": finite_or_none("R_rad_abs_glass_K_W"),
            "r34conv": finite_or_none("R_conv_annulus_K_W"),
            "r45": (
                math.log(float(g["D5"]) / float(g["D4"])) /
                (2.0 * math.pi * float(mats["glass"]["k"]) * dx)
                if has_glass else None
            ),
            "r56": finite_or_none("R_conv_external_K_W"),
            "r57": finite_or_none("R_rad_sky_K_W"),
        },
        "Q": {
            "q12": qfluid,
            "q23": qfluid,
            "q34rad": qradag,
            "q34conv": qconvag,
            "q45": qconvext + qradsky,
            "q56": qconvext,
            "q57": qradsky,
            "qsolar": qsolar,
            "qsupports": qsupports,
            "qdh": qdh,
        },
    }

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8" />
<style>
:root{{--ink:#111827;--muted:#64748b;--wire:#111827;--flow:#ef4444;--solar:#f59e0b;--border:#e2e8f0}}
*{{box-sizing:border-box}}
html,body{{margin:0;padding:0;background:transparent;font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;color:var(--ink)}}
.card{{background:#fff;border:1px solid var(--border);border-radius:18px;overflow:hidden;box-shadow:0 8px 24px rgba(15,23,42,.05)}}
.head{{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 16px 12px;border-bottom:1px solid #eef2f7;background:linear-gradient(180deg,#fff,#fbfdff)}}
.title{{font-size:16px;font-weight:730;letter-spacing:-.01em}}
.meta{{font-size:11px;color:var(--muted)}}
.scene{{padding:8px 10px 4px;background:#fff}}
svg{{width:100%;height:530px;display:block}}
.wire{{fill:none;stroke:var(--wire);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}}
.resistor{{fill:none;stroke:var(--wire);stroke-width:3.1;stroke-linecap:round;stroke-linejoin:round}}
.node{{fill:#111827;stroke:#fff;stroke-width:2}}
.node-halo{{fill:#fff;stroke:#dbe4ee;stroke-width:1.2}}
.node-name{{font-size:12px;font-weight:750;fill:#111827}}
.node-temp{{font-size:11px;fill:#64748b}}
.branch-label{{font-size:11px;font-weight:700;fill:#334155}}
.branch-value{{font-size:10px;fill:#64748b}}
.flow-line{{fill:none;stroke:var(--flow);stroke-width:2;stroke-linecap:round;opacity:.78}}
.flow-dot{{fill:var(--flow);filter:drop-shadow(0 0 2px rgba(239,68,68,.35))}}
.solar-line{{stroke:var(--solar);stroke-width:3;stroke-linecap:round}}
.solar-dot{{fill:var(--solar);filter:drop-shadow(0 0 3px rgba(245,158,11,.4))}}
.solar-label{{font-size:11px;font-weight:750;fill:#92400e}}
.legend{{display:flex;gap:18px;align-items:center;padding:0 16px 12px;font-size:11px;color:#64748b}}
.legend i{{display:inline-block;width:20px;height:3px;border-radius:999px;vertical-align:middle;margin-right:6px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;padding:0 16px 14px}}
.metric{{border:1px solid #e8edf3;border-radius:12px;padding:8px 10px;background:#fff}}
.k{{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.035em}}
.v{{font-size:14px;font-weight:740;margin-top:2px}}
@media(max-width:900px){{svg{{height:480px}}.summary{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="card">
  <div class="head"><div class="title">Circuito térmico · nodo axial {data['node']}</div><div class="meta">LAT {data['LAT']:.3f} h</div></div>
  <div class="scene"><svg id="circuit" viewBox="0 0 1120 560" preserveAspectRatio="xMidYMid meet"></svg></div>
  <div class="legend"><span><i style="background:#111827"></i>resistencia térmica</span><span><i style="background:#ef4444"></i>sentido del flujo</span><span><i style="background:#f59e0b"></i>aporte solar</span></div>
  <div class="summary">
    <div class="metric"><div class="k">Q solar / nodo</div><div class="v">{qsolar:.2f} W</div></div>
    <div class="metric"><div class="k">Absorbedor → HTF</div><div class="v">{qfluid:.2f} W</div></div>
    <div class="metric"><div class="k">ΔH axial HTF</div><div class="v">{qdh:.2f} W</div></div>
    <div class="metric"><div class="k">Pérdida exterior</div><div class="v">{(qconvext+qradsky+qsupports):.2f} W</div></div>
  </div>
</div>
<script>
const D={_js(data)}, svg=document.getElementById('circuit'), NS='http://www.w3.org/2000/svg';
function E(tag,a={{}}){{const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(a))e.setAttribute(k,v);return e;}}
function add(tag,a={{}},parent=svg){{const e=E(tag,a);parent.appendChild(e);return e;}}
function txt(x,y,text,cls,anchor='middle'){{const e=add('text',{{x,y,class:cls,'text-anchor':anchor}});e.textContent=text;return e;}}
function fmtR(v){{return (v===null||!Number.isFinite(v))?'∞':(Math.abs(v)>=1000?v.toExponential(2):v.toPrecision(4))+' K/W';}}
function fmtQ(v){{return `${{v.toFixed(2)}} W`;}}

const defs=add('defs');
const marker=add('marker',{{id:'arrowFlow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto-start-reverse'}},defs);
add('path',{{d:'M 0 0 L 10 5 L 0 10 z',fill:'#ef4444'}},marker);
const markerSolar=add('marker',{{id:'arrowSolar',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:5,markerHeight:5,orient:'auto'}},defs);
add('path',{{d:'M 0 0 L 10 5 L 0 10 z',fill:'#f59e0b'}},markerSolar);

function wire(x1,y1,x2,y2){{add('path',{{d:`M${{x1}},${{y1}} L${{x2}},${{y2}}`,class:'wire'}});}}
function resistorPath(x1,y1,x2,y2,teeth=7){{
  const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L,nx=-uy,ny=ux;
  const lead=Math.min(20,L*.14), amp=Math.min(13,L*.11), usable=L-2*lead;
  let d=`M ${{x1}} ${{y1}} L ${{x1+lead*ux}} ${{y1+lead*uy}}`;
  const pts=teeth*2;
  for(let i=1;i<=pts;i++){{const s=lead+usable*i/(pts+1),off=(i%2?amp:-amp);d+=` L ${{x1+s*ux+off*nx}} ${{y1+s*uy+off*ny}}`;}}
  d+=` L ${{x2-lead*ux}} ${{y2-lead*uy}} L ${{x2}} ${{y2}}`;return d;
}}
function resistor(x1,y1,x2,y2){{add('path',{{d:resistorPath(x1,y1,x2,y2),class:'resistor'}});}}
function node(x,y,name,temp,num){{
  add('circle',{{cx:x,cy:y,r:17,class:'node-halo'}});add('circle',{{cx:x,cy:y,r:6,class:'node'}});
  txt(x,y-29,name,'node-name');txt(x,y-15,`${{temp.toFixed(2)}} °C`,'node-temp');txt(x,y+34,`(${{num}})`,'node-temp');
}}
function flowArrow(x1,y1,x2,y2,q,offset=24){{
  const dx=x2-x1,dy=y2-y1,L=Math.hypot(dx,dy)||1,nx=-dy/L,ny=dx/L;
  let a=[x1+nx*offset,y1+ny*offset],b=[x2+nx*offset,y2+ny*offset];
  if(q<0){{const t=a;a=b;b=t;}}
  const p=add('path',{{d:`M ${{a[0]}} ${{a[1]}} L ${{b[0]}} ${{b[1]}}`,class:'flow-line','marker-end':'url(#arrowFlow)'}});
  const dot=add('circle',{{r:4,class:'flow-dot'}});
  add('animateMotion',{{dur:`${{Math.max(1.2,2.8-Math.min(Math.abs(q)/120,1.3))}}s`,repeatCount:'indefinite',path:p.getAttribute('d')}},dot);
}}
function labelBranch(x,y,name,r,q){{txt(x,y,name,'branch-label');txt(x,y+15,`R = ${{fmtR(r)}} · Q = ${{fmtQ(q)}}`,'branch-value');}}
function branch(a,b,name,r,q,offset=25,labelDy=-33){{resistor(...a,...b);flowArrow(...a,...b,q,offset);labelBranch((a[0]+b[0])/2,(a[1]+b[1])/2+labelDy,name,r,q);}}

const N={{n1:[80,300],n2:[220,300],n3:[380,300],n4:[610,300],n5:[760,300],n6:[1030,420],n7:[1030,180]}};

add('path',{{d:`M ${{N.n3[0]}} 65 L ${{N.n3[0]}} 255`,class:'solar-line','marker-end':'url(#arrowSolar)'}});
const sdot=add('circle',{{r:5,class:'solar-dot'}});
add('animateMotion',{{dur:'1.8s',repeatCount:'indefinite',path:`M ${{N.n3[0]}} 65 L ${{N.n3[0]}} 248`}},sdot);
txt(N.n3[0],45,'Radiación solar','solar-label');txt(N.n3[0],61,`Q = ${{fmtQ(D.Q.qsolar)}}`,'branch-value');

branch(N.n1,N.n2,'Convección interna',D.R.r12,D.Q.q12,27,-42);
branch(N.n2,N.n3,'Conducción absorbedor',D.R.r23,D.Q.q23,-28,47);

if(D.hasGlass){{
  wire(N.n3[0],N.n3[1],N.n3[0],180);wire(N.n4[0],N.n4[1],N.n4[0],180);
  wire(N.n3[0],N.n3[1],N.n3[0],420);wire(N.n4[0],N.n4[1],N.n4[0],420);
  branch([N.n3[0],180],[N.n4[0],180],'Radiación abs. → vidrio',D.R.r34rad,D.Q.q34rad,-24,-39);
  branch([N.n3[0],420],[N.n4[0],420],'Convección anular',D.R.r34conv,D.Q.q34conv,24,47);
  branch(N.n4,N.n5,'Conducción vidrio',D.R.r45,D.Q.q45,27,-42);
  wire(N.n5[0],N.n5[1],N.n5[0],180);wire(N.n5[0],N.n5[1],N.n5[0],420);
  branch([N.n5[0],180],N.n7,'Radiación → cielo',D.R.r57,D.Q.q57,-25,-39);
  branch([N.n5[0],420],N.n6,'Convección → ambiente',D.R.r56,D.Q.q56,25,47);
  node(...N.n1,'Tagua',D.T.Tagua,1);node(...N.n2,'Tabs,int',D.T.TabsI,2);node(...N.n3,'Tabs,ext',D.T.TabsE,3);
  node(...N.n4,'Tvid,int',D.T.TvidI,4);node(...N.n5,'Tvid,ext',D.T.TvidE,5);node(...N.n6,'Tamb',D.T.Tamb,6);node(...N.n7,'Tsky',D.T.Tsky,7);
}} else {{
  wire(N.n3[0],N.n3[1],N.n3[0],180);wire(N.n3[0],N.n3[1],N.n3[0],420);
  branch([N.n3[0],180],N.n7,'Radiación → cielo',D.R.r57,D.Q.q57,-25,-39);
  branch([N.n3[0],420],N.n6,'Convección → ambiente',D.R.r56,D.Q.q56,25,47);
  node(...N.n1,'Tagua',D.T.Tagua,1);node(...N.n2,'Tabs,int',D.T.TabsI,2);node(...N.n3,'Tabs,ext',D.T.TabsE,3);node(...N.n6,'Tamb',D.T.Tamb,6);node(...N.n7,'Tsky',D.T.Tsky,7);
}}
</script>
</body></html>"""
