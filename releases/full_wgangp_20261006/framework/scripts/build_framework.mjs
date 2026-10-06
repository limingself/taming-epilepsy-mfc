import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';
process.on('uncaughtException',error=>{console.error(error.message);process.exit(1);});

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const SKILL='C:/Users/LiMing/.codex/plugins/cache/openai-primary-runtime/presentations/26.921.10847/skills/presentations';
const PYTHON='C:/Users/LiMing/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/LiMing/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {finalizePresentation,resolvePresentationFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const DATA=JSON.parse(await fs.readFile(path.join(ROOT,'data_geometry.json'),'utf8'));
const FONT=resolvePresentationFont({fontFamily:'Arial'});
const MATH=resolvePresentationFont({fontFamily:'Cambria Math'});
const W=1800,H=1480;
const p=Presentation.create({slideSize:{width:W,height:H}});
const s=p.slides.add();s.background.fill='#FFFFFF';
const C={ink:'#23313E',muted:'#657282',blue:'#4C78B5',green:'#75944C',purple:'#7A629A',blueFill:'#EEF3FA',greenFill:'#EFF5E8',purpleFill:'#F3EEF8'};
const bounds=[];let n=0;
function shape(name,x,y,w,h,{fill='none',stroke='none',radius=0,width=1.5,geometry='rect',dashed=false}={}){
  const o=s.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h},fill,
    line:{fill:stroke,width:stroke==='none'?0:width,style:dashed?'dashed':'solid'},...(radius?{borderRadius:radius}:{})});
  bounds.push({name,x,y,w,h,type:geometry});return o;
}
function txt(name,x,y,w,h,value,{size=24,bold=false,color=C.ink,align='center',font=FONT,italic=false}={}){
  const o=shape(name,x,y,w,h,{geometry:'textbox'});o.text=value;
  o.text.style={typeface:font,fontSize:size,bold,italic,color,alignment:align,verticalAlignment:'middle',autoFit:'none',wrap:'none',insets:{top:0,right:0,bottom:0,left:0}};return o;
}
function box(name,x,y,w,h,{color=C.muted,fill='#FFFFFF',dashed=false}={}){
  return shape(name,x,y,w,h,{fill,stroke:color,radius:11,width:1.7,dashed});
}
function line(name,x1,y1,x2,y2,color,width=1.5,style='solid'){
  const x=Math.min(x1,x2),y=Math.min(y1,y2),w=Math.max(.2,Math.abs(x2-x1)),h=Math.max(.2,Math.abs(y2-y1));
  const o=s.shapes.add({name,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:color,width,style},
    customPaths:[{width:w,height:h,commands:[{moveTo:{x:x1-x,y:y1-y}},{lineTo:{x:x2-x,y:y2-y}}]}]});return o;
}
function anchor(x,y,name='anchor'){return shape(`${name}-${n++}`,x-.15,y-.15,.3,.3,{geometry:'ellipse'});}
function link(a,b,color,{from='bottom',to='top',kind='straight',dashed=false,arrow=true}={}){
  const c=s.shapes.connect(a,b,{kind,fromSide:from,toSide:to,line:{fill:color,width:2.2,style:dashed?'dashed':'solid'},...(arrow?{tail:{type:'triangle',width:'sm',length:'sm'}}:{})});
  c.bringToFront();return c;
}
function route(name,points,color,{dashed=false,arrow=true}={}){
  const anchors=points.map((a,i)=>anchor(a[0],a[1],`${name}-${i}`));
  for(let i=0;i<points.length-1;i++){
    const dx=points[i+1][0]-points[i][0],dy=points[i+1][1]-points[i][1];
    link(anchors[i],anchors[i+1],color,{from:Math.abs(dx)>Math.abs(dy)?(dx>0?'right':'left'):(dy>0?'bottom':'top'),to:Math.abs(dx)>Math.abs(dy)?(dx>0?'left':'right'):(dy>0?'top':'bottom'),dashed,arrow:arrow&&i===points.length-2});
  }
}
function pathSeries(name,x,y,w,h,points,color,width=1.65,style='solid',fill='none',closed=false){
  const commands=points.map((a,i)=>i?{lineTo:{x:a[0]*w,y:a[1]*h}}:{moveTo:{x:a[0]*w,y:a[1]*h}});
  if(closed)commands.push({close:{}});
  const o=s.shapes.add({name,geometry:'custom',position:{left:x,top:y,width:w,height:h},fill,
    line:{fill:color,width:color==='none'?0:width,style},customPaths:[{width:w,height:h,commands}]});
  bounds.push({name,x,y,w,h,type:'editable-data-path',vertices:points.length});return o;
}
function math(name,x,y,w,tokens,{size=29,color=C.ink}={}){
  // Each symbol and each sub/superscript is a separate editable text object.
  const chars=a=>[...String(a)].length;
  const prepared=tokens.map(a=>typeof a==='string'?{base:a,literal:true}:a).map(a=>{
    const bw=a.width??Math.max(size*.49,chars(a.base)*size*(a.literal?.53:.58));
    const sw=Math.max(chars(a.sub||''),chars(a.sup||''))*size*.32;
    return {...a,bw,tw:bw+sw+(a.gap??size*.12)};
  });
  const total=prepared.reduce((v,a)=>v+a.tw,0);let px=x+(w-total)/2;
  if(total>w)throw new Error(`Math exceeds slot ${name}: ${total} > ${w}`);
  prepared.forEach((a,i)=>{
    const isVectorOrMatrix=!a.literal&&['x','z','Δz','A','g','v','r','Q','φ','S','u','B','H'].includes(a.base);
    txt(`${name}-${i}-base`,px,y,a.bw,size*1.22,a.base,{size,color,font:MATH,bold:isVectorOrMatrix});
    if(a.sub)txt(`${name}-${i}-sub`,px+a.bw-1,y+size*.70,a.tw-a.bw,size*.65,a.sub,{size:size*.61,color,font:MATH,align:'left'});
    if(a.sup)txt(`${name}-${i}-sup`,px+a.bw-1,y-size*.10,a.tw-a.bw,size*.67,a.sup,{size:size*.61,color,font:MATH,align:'left'});
    if(a.accent==='bar')line(`${name}-${i}-bar`,px+2,y+size*.10,px+a.bw-1,y+size*.10,color,1.5);
    if(a.accent==='tilde')pathSeries(`${name}-${i}-tilde`,px+1,y+size*.05,Math.max(10,a.bw-2),5.5,[[0,.75],[.18,.22],[.36,.17],[.60,.72],[.80,.78],[1,.22]],color,1.35);
    if(a.accent==='hat')pathSeries(`${name}-${i}-hat`,px+2,y+size*.06,Math.max(10,a.bw-4),5.0,[[0,.82],[.5,0],[1,.82]],color,1.35);
    px+=a.tw;
  });
}
function clock(name,x,y,size,color){
  shape(`${name}-face`,x,y,size,size,{geometry:'ellipse',stroke:color,width:1.8,fill:'#FFFFFF'});
  line(`${name}-hand-hour`,x+size/2,y+size/2,x+size/2,y+size*.22,color,1.6);
  line(`${name}-hand-minute`,x+size/2,y+size/2,x+size*.73,y+size*.59,color,1.6);
}
function knockout(name,x,y,color,style='solid'){
  shape(name,x-6,y-7,12,14,{fill:'#FFFFFF'});line(`${name}-through`,x,y-7,x,y+7,color,2.2,style);
}

// Preserve the reference's three-column design, colors and evidence.
const cols=[{x:54,c:C.blue,f:C.blueFill,n:'1',title:'Patient-specific\nnetwork selection',subtitle:'Patient-specific iEEG'},
  {x:640,c:C.green,f:C.greenFill,n:'2',title:'Stochastic\nGraph–RC surrogate',subtitle:'Training-fitted stochastic plant'},
  {x:1226,c:C.purple,f:C.purpleFill,n:'3',title:'Mean–deviation\nfeedback',subtitle:'Individual preictal reference'}];
cols.forEach(a=>{
  shape(`column-${a.n}`,a.x,30,520,1260,{stroke:a.c,fill:'#FFFFFF',radius:35,width:2.3});
  shape(`stage-number-${a.n}`,a.x+22,56,41,41,{geometry:'ellipse',stroke:a.c,width:1.8,fill:'#FFFFFF'});
  txt(`stage-number-text-${a.n}`,a.x+22,56,41,41,a.n,{size:29,bold:true,color:a.c});
  txt(`column-title-${a.n}`,a.x+70,50,414,78,a.title,{size:29,bold:true});
  box(`column-subtitle-${a.n}`,a.x+30,143,460,38,{color:a.c,fill:a.f});
  txt(`subtitle-text-${a.n}`,a.x+30,143,460,38,a.subtitle,{size:22});
});

// I. Actual traces, actual selection graph and actual A_I adjacency.
const recordedBox={x:86,y:200,w:456,h:456/DATA.original_plot_aspects.recorded};
DATA.recorded.series.forEach((a,i)=>pathSeries(`recorded-${a.name}`,recordedBox.x,recordedBox.y,recordedBox.w,recordedBox.h,a.points,C.blue,1.5));
const pre=box('preprocessing',94,334,440,78,{color:C.blue,fill:'#F8FAFD'});
txt('preprocessing-label',101,339,426,49,'Referencing and filtering\nTraining-set standardization',{size:23});
math('standardized-x',110,377,408,[{base:'x',sub:'t',accent:'tilde'}],{size:25,color:C.blue});
route('trace-to-pre',[[314,303],[314,328]],C.blue);
route('pre-to-graph',[[314,417],[314,440]],C.blue);
txt('plv-title',98,451,432,53,'Connected multiband\nPLV network',{size:25,bold:true});
const gx=218,gy=626,r=90;
DATA.edges.forEach((e,i)=>{const a=DATA.nodes[e.source],b=DATA.nodes[e.target];line(`PLV-edge-${i}`,gx+a.x*r,gy-a.y*r,gx+b.x*r,gy-b.y*r,'#B7C4D4',.8+.8*e.weight);});
DATA.nodes.forEach(a=>shape(`contact-${a.channel}`,gx+a.x*r-5,gy-a.y*r-5,10,10,{geometry:'ellipse',fill:a.selected?C.blue:'#D5DEE8',stroke:'#FFFFFF',width:.7}));
const mx=366,my=545,cell=4.4;
DATA.adjacency.forEach((row,i)=>row.forEach((value,j)=>{
  shape(`A-I-row-${i}-col-${j}-value-${value}`,mx+j*cell,my+i*cell,cell+.02,cell+.02,{fill:DATA.adjacency_colors[i][j]});
}));
txt('adjacency-label',343,709,203,28,'Adjacency',{size:23});
math('adjacency-A-I',343,740,203,[{base:'A',sub:'I'}],{size:28,color:C.blue});
const central=box('weighted-centrality',94,793,440,92,{color:C.blue,fill:C.blueFill});
txt('centrality-text',104,796,420,86,'Weighted centrality\nStrength + betweenness\n+ eigenvector',{size:24});
route('graph-to-centrality',[[314,741],[314,786]],C.blue);
txt('select-label',94,934,440,35,'Select virtual-input contacts',{size:24,bold:true});
route('centrality-to-selection',[[314,891],[314,926]],C.blue);
DATA.nodes.forEach((a,i)=>shape(`mask-cell-${a.channel}`,104+i*11.72,998,10.4,26,{fill:a.selected?C.blue:'#E0E6EE'}));
const mask=box('selected-actuator-mask',94,1130,440,103,{color:C.blue,fill:'#F8FAFD'});
txt('mask-title',104,1138,420,29,'Actuator mask',{size:25,bold:true});
math('S-A',106,1170,416,[{base:'S',sub:'𝒜'}],{size:29,color:C.blue});
txt('mask-support',104,1204,420,25,'Virtual-input support',{size:22});
route('selected-to-mask',[[314,1033],[314,1124]],C.blue);

// II. Separate fitted graph A_II, concatenated input v_t, leaky reservoir r_t.
const state=box('PCA-state-delay-history',680,211,204,122,{color:C.green,fill:'#F9FBF6'});
txt('state-heading',686,217,192,30,'PCA state',{size:23});
math('latent-state-z',688,249,188,[{base:'z',sub:'t'}],{size:29,color:C.green});
txt('delay-label',717,283,151,22,'Delay history',{size:20});
clock('delay-clock',694,282,22,C.green);
math('delay-coordinates',686,304,192,[{base:'z',sub:'t−1'},',',{base:'z',sub:'t−8'},',',{base:'z',sub:'t−32'}],{size:19,color:C.green});
const graph=box('separate-fitted-graph',916,211,204,122,{color:C.green,fill:'#F9FBF6'});
txt('separate-graph-heading',924,217,188,44,'Separate fitted\nPLV graph',{size:22});
math('A-II',926,262,184,[{base:'A',sub:'II'}],{size:30,color:C.green});
math('graph-feature-g',926,298,184,[{base:'g',sub:'t'}],{size:23,color:C.green});
route('preprocessed-data-to-PCA',[[540,373],[606,373],[606,268],[674,268]],C.blue);
const features=box('state-graph-delays',680,389,440,88,{color:C.green,fill:C.greenFill});
txt('feature-title',688,395,424,32,'State + graph + delays',{size:25,bold:true});
math('feature-input-v',688,429,424,[{base:'v',sub:'t'}],{size:29,color:C.green});
link(state,features,C.green,{kind:'elbow'});link(graph,features,C.green,{kind:'elbow'});
const reservoir=box('leaky-reservoir',680,528,440,89,{color:C.green});
txt('reservoir-heading',688,534,424,32,'Leaky reservoir',{size:25,bold:true});
math('reservoir-state-r',688,568,424,[{base:'r',sub:'t'}],{size:29,color:C.green});
link(features,reservoir,C.green);
const drift=box('fitted-drift',680,674,204,94,{color:C.green,fill:C.greenFill});
txt('fitted-drift-label',686,679,192,30,'Fitted drift',{size:24});
math('delta-z-hat',688,715,188,[{base:'Δz',sub:'t+1',accent:'hat'}],{size:29,color:C.green});
const noise=box('conditional-innovation-covariance',916,674,204,94,{color:C.green,fill:C.greenFill});
txt('conditional-innovation-label',922,679,192,45,'Conditional innovation\ncovariance',{size:20});
math('conditional-Q',922,730,192,['Q(',{base:'φ',sub:'t',accent:'tilde'},')'],{size:25,color:C.green});
link(reservoir,drift,C.green,{kind:'elbow'});link(reservoir,noise,C.green,{kind:'elbow'});
const recurrence=box('frozen-recurrence',680,824,440,75,{color:C.green,fill:'#F9FBF6'});
txt('recurrence-heading',688,829,424,30,'Frozen stochastic recurrence',{size:24,bold:true});
math('frozen-Phi',688,862,424,[{base:'Φ',sub:'θ'}],{size:29,color:C.green});
link(drift,recurrence,C.green,{kind:'elbow'});link(noise,recurrence,C.green,{kind:'elbow'});
const freeBox={x:682,y:948,w:436,h:436/DATA.original_plot_aspects.free_paths};
const greens=['#A8BD86','#91AA6D','#79994F','#B3C695'];
DATA.free_paths.series.forEach((a,i)=>pathSeries(`free-${a.name}`,freeBox.x,freeBox.y,freeBox.w,freeBox.h,a.points,greens[i],1.65));
route('recurrence-to-paths',[[900,905],[900,940]],C.green);
txt('autonomous-paths-label',680,1083,440,30,'Autonomous particle paths',{size:25,bold:true});
const empirical=box('32-particle-empirical-law',680,1145,440,72,{color:C.green,fill:C.greenFill});
txt('empirical-heading',688,1154,424,24,'32-particle empirical law',{size:22});
math('empirical-mu',688,1180,424,[{base:'μ',sub:'k',sup:'M'},'=',{base:'1/M'},{base:'Σ',sub:'m=1',sup:'M'},{base:'δ',sub:'xₖ⁽ᵐ⁾'}],{size:24,color:C.green});

// III. Reference and decoded predicted statistics supply both actor branches.
const refBox={x:1267,y:200,w:438,h:438/DATA.original_plot_aspects.reference};
const rs=DATA.reference.series[0];
pathSeries('reference-density-fill',refBox.x,refBox.y,refBox.w,refBox.h,[[rs.points[0][0],DATA.reference.baseline_zero],...rs.points,[rs.points.at(-1)[0],DATA.reference.baseline_zero]],'none',0,'solid','#E8F1EA',true);
pathSeries('reference-density-curve',refBox.x,refBox.y,refBox.w,refBox.h,rs.points,'#3C8D62',2.5);
math('reference-mu',1267,302,438,[{base:'μ',sup:'ref'}],{size:29,color:'#3C8D62'});
const joint=shape('reference-and-predicted-junction',1482,351,8,8,{geometry:'ellipse',fill:C.purple});
route('reference-to-junction',[[1486,339],[1486,345]],C.purple);
const common=box('common-actor-branch',1288,398,182,94,{color:C.purple,fill:C.purpleFill});
txt('common-branch-label',1294,402,170,47,'Common input\nMean + spread',{size:23});
math('common-u',1294,450,170,[{base:'u',sub:'k',accent:'bar'}],{size:27,color:C.purple});
const deviation=box('centered-deviation-branch',1502,398,182,94,{color:C.purple,fill:C.purpleFill});
txt('deviation-branch-label',1508,402,170,47,'Centered\ncorrection',{size:23});
math('centered-u',1508,450,170,[{base:'u',sub:'k',sup:'(m)',accent:'tilde'}],{size:27,color:C.purple});
link(joint,common,C.purple,{kind:'elbow'});link(joint,deviation,C.purple,{kind:'elbow'});
math('combined-input-policy',1308,551,356,[{base:'u',sub:'k',sup:'(m)'},'=',{base:'u',sub:'k',accent:'bar'},'+',{base:'u',sub:'k',sup:'(m)',accent:'tilde'}],{size:33,color:C.purple});
route('common-to-combined',[[1379,498],[1379,523],[1454,542]],C.purple);
route('centered-to-combined',[[1593,498],[1593,523],[1520,542]],C.purple);
const bounded=box('bounded-virtual-input',1288,624,396,69,{color:C.purple,fill:'#FBF9FD'});
txt('bounded-input-heading',1296,628,380,28,'Bounded virtual input',{size:24,bold:true});
math('input-bound',1296,657,380,['∥',{base:'u',sub:'k',sup:'(m)'},{base:'∥',sub:'∞'},'≤ 1.8'],{size:24,color:C.purple});
route('combined-to-bound',[[1486,596],[1486,618]],C.purple);
const input=box('heat-kernel-actuator-input-map',1288,742,396,78,{color:C.purple,fill:'#FBF9FD'});
txt('input-map-heading',1296,747,380,27,'Heat kernel + actuator mask',{size:23});
math('B-G-definition',1296,778,380,[{base:'B',sub:'G'},'=',{base:'H',sub:'τ',sup:'T'},{base:'S',sub:'𝒜',sup:'T'}],{size:27,color:C.purple});
link(bounded,input,C.purple);
const plant=box('same-frozen-Graph-RC-plant',1288,872,396,75,{color:C.purple,fill:C.purpleFill});
txt('plant-heading',1296,877,380,29,'Frozen Graph–RC plant',{size:24,bold:true});
math('model-virtual-input',1296,903,380,[{base:'α',sub:'u'},{base:'B',sub:'G'},{base:'u',sub:'k',sup:'(m)'}],{size:28,color:C.purple});
link(input,plant,C.purple);
route('same-recurrence-to-plant',[[1126,861],[1186,861],[1186,910],[1282,910]],C.green);
route('A-II-to-input-mapping',[[1126,272],[1172,272],[1172,766],[1282,766]],C.green);
route('S-A-to-input-mapping',[[314,1239],[314,1314],[1200,1314],[1200,802],[1282,802]],C.blue);
txt('mask-route-label',679,1290,440,21,'Selected virtual-input support',{size:20,color:C.blue});
knockout('mask-model-independent-crossover',1200,910,C.blue);
route('predicted-state-law-feedback',[[1690,907],[1723,907],[1723,355],[1498,355]],C.purple);
txt('predicted-feedback-label',1504,323,200,25,'Predicted state / law',{size:20,color:C.purple});
const styles=['solid','dashed','dashDot','solid'];
const curveColors=['#30363B','#9AA3AC','#3C8D62','#2166AC'];
const names=['Recorded','Free','Reference','Controlled'];
names.forEach((label,i)=>{
  const x=1299+(i%2)*202,y=978+Math.floor(i/2)*30;
  line(`curve-key-${i}`,x,y,x+29,y,curveColors[i],2.2,styles[i]);
  txt(`curve-key-name-${i}`,x+37,y-13,157,26,label,{size:22,color:curveColors[i],align:'left'});
});
const densityBox={x:1280,y:1041,w:412,h:412/DATA.original_plot_aspects.densities};
DATA.densities.series.forEach((a,i)=>pathSeries(`density-${names[i]}`,densityBox.x,densityBox.y,densityBox.w,densityBox.h,a.points,curveColors[i],i===3?2.5:1.9,styles[i]));
txt('occupation-law-label',1278,1200,416,28,'Marginal occupation laws',{size:24,bold:true});
const train=box('law-loss-WGAN-training-only',1288,1243,396,35,{color:C.purple,dashed:true});
txt('training-only-label',1296,1245,380,31,'Law loss + WGAN-GP (training)',{size:21});
route('training-only-policy-update',[[1282,1260],[1253,1260],[1253,572],[1300,572]],C.purple,{dashed:true});
for(const y of [766,802,910])knockout(`training-independent-crossover-${y}`,1253,y,C.purple,'dashed');

// Shared evaluation band retains the patient-specific protocol roles.
txt('evaluation-heading',74,1350,1652,38,'Predictive fidelity and model-law steering',{size:28,bold:true});
const patients=[['HUP060','Development / adaptation','13 / 36 direct inputs'],['HUP065','Post hoc amended analysis','32 / 64 direct inputs'],['HUP080','Post hoc amended analysis','76 / 96 direct inputs']];
patients.forEach((a,i)=>{
  box(`case-panel-${i}`,cols[i].x,1410,520,56,{color:cols[i].c});
  txt(`case-ID-${i}`,cols[i].x+8,1414,504,23,`${a[0]}   ${a[2]}`,{size:23,bold:true,color:cols[i].c});
  txt(`case-role-${i}`,cols[i].x+8,1438,504,23,a[1],{size:21,color:C.muted});
});

s.speakerNotes.textFrame.setText([
  'The connected multiband Part-I PLV graph A_I supplies the virtual-actuator mask S_A. The displayed graph and heatmap use the HUP060 source data: 36 modeled contacts, 63 undirected edges and 13 selected contacts. This selection graph is distinct from the separately fitted Part-II graph A_II.',
  'Part II: standardized electrode state x_tilde_t is reduced to z_t. The concatenated feature input v_t includes z_t, lagged coordinates, and graph features g_t=P A_II x_tilde_t. The leaky reservoir has state r_t. The readout feature phi_t concatenates reservoir state r_t and all terms of v_t and is then standardized to phi_tilde_t. The drift estimate is Delta z_hat_(t+1)=B phi_tilde_t+b. Conditional Q(phi_tilde_t)=D(phi_tilde_t) R D(phi_tilde_t) is an innovation covariance, not a noise sample. Actual innovations are kappa_cal D(phi_tilde_t) R^(1/2) epsilon_t^(m). The frozen recurrence is Phi_theta.',
  'Four paths illustrate zero-based particles 0–3 at RPFa3. The empirical law uses M=32 particles: mu_k^M=(1/M) sum_(m=1)^M delta_(x_k^(m)). Recorded traces use the source display offsets 0, 3.5 and 7. All mini-plots preserve the original Python figure’s data transforms and aspect ratios.',
  'Part III: the common command u_bar_k and centered correction u_tilde_k^(m) form u_k^(m)=u_bar_k+u_tilde_k^(m), with sum_m u_tilde_k^(m)=0 and infinity norm at most 1.8. Both fixed reference buffers and decoded one-step predicted state/law supply the two actor branches. The reference mini-plot depicts the saved validation reference density, while actor training uses R_fit buffers.',
  'The input map is B_G=H_tau^T S_A^T, with H_tau=exp(-tau_G L_rw) and L_rw=I-D_II^(-1) A_II. The injected model input is alpha_u B_G u_k^(m). Green routing denotes the separate Part-II graph and frozen recurrence. Blue routing denotes selected virtual-input support. The heat kernel is an input mapping component, while the latent/reservoir recurrence provides dynamical coupling.',
  'The four density curves are time-pooled marginal occupation laws at RPFa3. They are not instantaneous particle laws. Recorded is charcoal, free is grey dashed, reference is green dash-dot, controlled is blue. The historical source label Interictal reference is displayed as Preictal reference consistently with the verified manuscript intervals.',
  'WGAN-GP and the law loss train the actor pi_vartheta using critic D_omega. The dashed route updates the combined policy and is separate from inference-time state/law feedback.',
  'HUP060 is the development/adaptation case (13/36 direct inputs). HUP065 uses an expanded top32/64 actuator mask and HUP080 retains 76/96. Both current external controller results are post hoc amendments of previously revealed outer seizures, not pristine locked validation. Full WGAN-GP retains the frozen Graph-RC plant and structured mean-deviation actor with empirical-law penalties. WGAN-GP has been present from the first fresh actor update; subsequent parameter adaptations are separately documented.',
  'The blue Controlled mini-curve uses the HUP060 fresh primary checkpoint selected at actor update 700, SHA256 980952488c536b76cfc51fa5d979243e0a69be4cba1cbf36aacdc094e7b29ab0. Its plotted source is source_data_control_densities.csv from the current control figure. The three other density curves, graph, heatmap, recorded traces and free paths are unchanged; no new density fit or synthetic data is used.',
  'Sources: chaos_methods.tex and actor_implementation_recipe.tex in chaos_revision_20261005_round3, plus supporting_materials/HUP060/output/part1/source_data/figure_01 and supporting_materials/HUP060/output/part3/source_data/figures_06_08. The actual adjacency source file is hup060_adjacency_matrix.csv. Source data and parameter values are frozen.',
].join('\n\n'));
const outside=bounds.filter(a=>a.x<0||a.y<0||a.x+a.w>W||a.y+a.h>H);
if(outside.length)throw new Error(`Outside slide: ${JSON.stringify(outside)}`);
await fs.writeFile(path.join(ROOT,'qa','declared_layout.json'),JSON.stringify({width:W,height:H,bounds,outside,plotBoxes:{recorded:recordedBox,reference:refBox,free:freeBox,density:densityBox},native_diagram:true,native_math:true,heatmap_cells:1296},null,2));
const preview=await p.export({slide:s,format:'png',scale:1});
await fs.writeFile(path.join(ROOT,'qa','draft_preview.png'),new Uint8Array(await preview.arrayBuffer()));
const layout=await s.export({format:'layout'});
await fs.writeFile(path.join(ROOT,'qa','slide.layout.json'),await layout.text());
await(await PresentationFile.exportPptx(p)).save(path.join(ROOT,'qa','candidate.pptx'));
if(!process.argv.includes('--final')){
  console.log(JSON.stringify({draft:true,preview:path.join(ROOT,'qa','draft_preview.png'),fonts:[FONT,MATH],shapes:bounds.length}));
  process.exit(0);
}
const final=path.join(ROOT,'output','Framework_full_WGANGP_all_cases_20261006.pptx');
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:path.join(ROOT,'qa','candidate.pptx'),finalPath:final,pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[],requiredNativeTableOwnerSlides:[],
  fontPolicy:{basis:'design',families:[FONT,MATH]},verifyArtifactToolImport:true,
  receiptPath:path.join(ROOT,'qa','finalization_receipt_v2.json')});
const imported=await PresentationFile.importPptx(await FileBlob.load(final));
const finalPreview=await imported.export({slide:imported.slides.items[0],format:'png',scale:1});
await fs.writeFile(path.join(ROOT,'output','Framework_math_heatmap_preview_20261005.png'),new Uint8Array(await finalPreview.arrayBuffer()));
console.log(JSON.stringify({final,fonts:[FONT,MATH],slides:1,heatmapCells:1296,status:result.status||'finalized'}));
