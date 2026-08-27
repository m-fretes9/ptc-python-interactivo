
const D={"hasGlass": false, "node": 1, "LAT": 16.0, "T": {"Tagua": 26.152459201325485, "TabsI": 92.9622659689415, "TabsE": 92.9622659689415, "TvidI": 25.0, "TvidE": 25.0, "Tamb": 25.0, "Tsky": 9.395194943800902}, "R": {"r12": 0.7221418768485323, "r23": 0.00018381272868750226, "r34rad": 0.0, "r34conv": 0.0, "r45": null, "r56": 2.7804609152879927, "r57": 6.42292125730376}, "Q": {"q12dir": -92.50440552698993, "q23dir": -92.50440552698993, "q12": 92.50440552698993, "q23": 92.50440552698993, "q34radDir": 0.0, "q34convDir": 0.0, "q45Dir": 37.45356323793108, "q56Dir": 24.44280572161977, "q57Dir": 13.01075751631131, "q34rad": 0.0, "q34conv": 0.0, "q45": 37.45356323793108, "q56": 24.44280572161977, "q57": 13.01075751631131, "qsolar": 129.95837708333335, "qsupports": 0.0, "qdh": 92.50435288231608}}, svg=document.getElementById('circuit'), NS='http://www.w3.org/2000/svg';
function E(tag,a={}){const e=document.createElementNS(NS,tag);for(const [k,v] of Object.entries(a))e.setAttribute(k,v);return e;}
function add(tag,a={},parent=svg){const e=E(tag,a);parent.appendChild(e);return e;}
function txt(x,y,text,cls,anchor='middle'){const e=add('text',{x,y,class:cls,'text-anchor':anchor});e.textContent=text;return e;}
function fmtR(v){return (v===null||!Number.isFinite(v))?'∞':(Math.abs(v)>=1000?v.toExponential(2):v.toPrecision(4))+' K/W';}
function fmtQ(v){return `${Math.abs(v).toFixed(2)} W`;}
function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t];}

const defs=add('defs');
const marker=add('marker',{id:'arrowFlow',viewBox:'0 0 10 10',refX:8.5,refY:5,markerWidth:4.4,markerHeight:4.4,orient:'auto'},defs);
add('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#ff5a4f'},marker);
const markerSolar=add('marker',{id:'arrowSolar',viewBox:'0 0 10 10',refX:8.5,refY:5,markerWidth:4.4,markerHeight:4.4,orient:'auto'},defs);
add('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#f59e0b'},markerSolar);

function wire(a,b){add('path',{d:`M ${a[0]} ${a[1]} L ${b[0]} ${b[1]}`,class:'wire'});}
function resistorPath(a,b,teeth=5){
  const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L,nx=-uy,ny=ux;
  const amp=Math.min(7,L*.12), steps=teeth*2;
  let d=`M ${a[0]} ${a[1]}`;
  for(let i=1;i<=steps;i++){const s=L*i/(steps+1),off=(i%2?amp:-amp);d+=` L ${a[0]+s*ux+off*nx} ${a[1]+s*uy+off*ny}`;}
  d+=` L ${b[0]} ${b[1]}`;return d;
}
function resistor(a,b){add('path',{d:resistorPath(a,b),class:'resistor'});}
function node(p,name,temp,num,anchor='middle'){
  add('circle',{cx:p[0],cy:p[1],r:4.5,class:'node'});
  const dx=anchor==='start'?10:anchor==='end'?-10:0;
  txt(p[0]+dx,p[1]-19,name,'node-name',anchor);txt(p[0]+dx,p[1]-7,`${temp.toFixed(2)} °C`,'node-temp',anchor);txt(p[0],p[1]+22,`(${num})`,'node-temp');
}
function shortFlow(a,b,qDir,side=1){
  const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1,ux=dx/L,uy=dy/L,nx=-uy/L*L,ny=ux/L*L;
  const mid=mix(a,b,.5), half=Math.min(24,L*.16), off=16*side;
  let s=[mid[0]-ux*half+(-uy)*off,mid[1]-uy*half+(ux)*off];
  let e=[mid[0]+ux*half+(-uy)*off,mid[1]+uy*half+(ux)*off];
  if(qDir<0){const t=s;s=e;e=t;}
  const path=`M ${s[0]} ${s[1]} L ${e[0]} ${e[1]}`;
  add('path',{d:path,class:'flow','marker-end':'url(#arrowFlow)'});
  if(Math.abs(qDir)>1e-9){const dot=add('circle',{r:2.4,class:'flow-dot'});add('animateMotion',{dur:'1.9s',repeatCount:'indefinite',path},dot);}
}
function branch(a,b,name,r,qDir,qDisplay,side=1){
  const rs=mix(a,b,.38), re=mix(a,b,.62);
  wire(a,rs);resistor(rs,re);wire(re,b);shortFlow(rs,re,qDir,side);
  const mid=mix(rs,re,.5), dx=re[0]-rs[0],dy=re[1]-rs[1],L=Math.hypot(dx,dy)||1,nx=-dy/L,ny=dx/L;
  const labelOff=-25*side;
  txt(mid[0]+nx*labelOff,mid[1]+ny*labelOff,name,'branch-name');
  txt(mid[0]+nx*labelOff,mid[1]+ny*labelOff+13,`R = ${fmtR(r)} · Q = ${fmtQ(qDisplay)}`,'branch-value');
}

const N={n1:[78,250],n2:[245,250],n3:[412,250],n4:[610,250],n5:[765,250],n6:[1080,355],n7:[1080,145]};

// Solar input at absorber outer surface.
add('path',{d:`M ${N.n3[0]} 45 L ${N.n3[0]} 224`,class:'solar','marker-end':'url(#arrowSolar)'});
const sdot=add('circle',{r:2.8,fill:'#f59e0b'});add('animateMotion',{dur:'1.8s',repeatCount:'indefinite',path:`M ${N.n3[0]} 45 L ${N.n3[0]} 216`},sdot);
txt(N.n3[0],28,'Radiación solar','solar-label');txt(N.n3[0],40,`Q = ${fmtQ(D.Q.qsolar)}`,'branch-value');

// Positive Qfluid from the solver is absorber -> HTF, hence negative direction
// on branches drawn geometrically from HTF (left) to absorber (right).
branch(N.n1,N.n2,'Convección interna',D.R.r12,D.Q.q12dir,D.Q.q12,1);
branch(N.n2,N.n3,'Conducción absorbedor',D.R.r23,D.Q.q23dir,D.Q.q23,-1);

if(D.hasGlass){
  const A=[N.n3[0],145], B=[N.n4[0],145], C=[N.n3[0],355], Dn=[N.n4[0],355];
  wire(N.n3,A);wire(N.n4,B);wire(N.n3,C);wire(N.n4,Dn);
  branch(A,B,'Radiación abs. → vidrio',D.R.r34rad,D.Q.q34radDir,D.Q.q34rad,1);
  branch(C,Dn,'Convección anular',D.R.r34conv,D.Q.q34convDir,D.Q.q34conv,-1);
  branch(N.n4,N.n5,'Conducción vidrio',D.R.r45,D.Q.q45Dir,D.Q.q45,1);
  const E7=[N.n5[0],145], E6=[N.n5[0],355];wire(N.n5,E7);wire(N.n5,E6);
  branch(E7,N.n7,'Radiación → cielo',D.R.r57,D.Q.q57Dir,D.Q.q57,1);
  branch(E6,N.n6,'Convección → ambiente',D.R.r56,D.Q.q56Dir,D.Q.q56,-1);
  node(N.n1,'Tagua',D.T.Tagua,1,'start');node(N.n2,'Tabs,int',D.T.TabsI,2);node(N.n3,'Tabs,ext',D.T.TabsE,3);
  node(N.n4,'Tvid,int',D.T.TvidI,4);node(N.n5,'Tvid,ext',D.T.TvidE,5);node(N.n6,'Tamb',D.T.Tamb,6,'end');node(N.n7,'Tsky',D.T.Tsky,7,'end');
} else {
  const A=[N.n3[0],145], C=[N.n3[0],355];wire(N.n3,A);wire(N.n3,C);
  branch(A,N.n7,'Radiación → cielo',D.R.r57,D.Q.q57Dir,D.Q.q57,1);
  branch(C,N.n6,'Convección → ambiente',D.R.r56,D.Q.q56Dir,D.Q.q56,-1);
  node(N.n1,'Tagua',D.T.Tagua,1,'start');node(N.n2,'Tabs,int',D.T.TabsI,2);node(N.n3,'Tabs,ext',D.T.TabsE,3);node(N.n6,'Tamb',D.T.Tamb,6,'end');node(N.n7,'Tsky',D.T.Tsky,7,'end');
}
