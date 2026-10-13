"use strict";
const $ = id => document.getElementById(id);
const colors = ["#087e86", "#d97526", "#586dc0", "#a45491", "#359468", "#b79632", "#bb5c5b", "#62859e"];
const groups = ["overview", "tracking", "feet", "energy", "recovery", "diagnostics", "all"];
const text = {
 zh: {brand:"训练看板",subtitle:"TQBot2 · TensorBoard 双语视图",runs:"训练记录",records:"组记录",runSearch:"搜索训练名称或日期",metricSearch:"搜索指标：课程、力矩、滑移…",eyebrow:"实时训练监测",refresh:"立即刷新",export:"导出当前数据",smoothing:"平滑",axis:"横轴",off:"关闭",rounds:"轮",iteration:"迭代编号",relative:"相对起点",time:"训练时长",hours:"小时",auto:"自动刷新",recent:"最近 500 轮",recentTime:"最近 30 分钟",reset:"显示全部",loading:"正在加载日志曲线…",empty:"未找到匹配指标。可用中文名称或原始标签搜索。",select:"请选择至少一组训练记录。",running:"训练中",completed:"已完成",stopped:"已停止",live:"每 20 秒更新",paused:"自动刷新已暂停",updated:"更新于",compared:"已选",count:"组训练",max:"最多选择 8 组训练记录。",higher:"越高越好",lower:"越低越好",notlogged:"无该指标",friction:"普通足端摩擦下限",summary:"训练总览",step:"迭代",raw:"原始",smoothed:"平滑",resetTip:"双击恢复全范围",failed:"读取日志失败，请稍后刷新。",warning:"所选训练的初始化来源或摩擦范围不同，曲线对比不能作为单因素归因。",sidefoot:"勾选记录即可对比，最多 8 组。颜色与曲线标记一致。页面仅查看日志。",help:"悬停查看数值；拖选区间可放大所有图，双击恢复。浅线为原始数据，深线为平滑数据。",footer:"迭代编号从 0 开始；恢复训练保留检查点编号。课程等级为全部环境均值，日志未记录坡面单独等级。训练曲线不等于固定工况性能测试。",overview:"训练总览",tracking:"指令跟踪",feet:"足端与步态",energy:"能耗与关节",recovery:"摔倒恢复",diagnostics:"训练诊断",all:"全部指标"},
 en: {brand:"Training Board",subtitle:"TQBot2 · Bilingual TensorBoard view",runs:"Training runs",records:"runs",runSearch:"Search run name or date",metricSearch:"Search: curriculum, torque, slip…",eyebrow:"LIVE TRAINING MONITOR",refresh:"Refresh now",export:"Export visible data",smoothing:"Smoothing",axis:"X axis",off:"Off",rounds:"iterations",iteration:"Iteration",relative:"Relative start",time:"Training time",hours:"hours",auto:"Auto refresh",recent:"Last 500 iterations",recentTime:"Last 30 minutes",reset:"Show all",loading:"Loading training curves…",empty:"No matching metrics. Search a translated name or an original tag.",select:"Select at least one training run.",running:"Running",completed:"Completed",stopped:"Stopped",live:"Updates every 20s",paused:"Auto refresh paused",updated:"Updated",compared:"Comparing",count:"runs",max:"Select up to 8 training runs.",higher:"Higher is better",lower:"Lower is better",notlogged:"Not logged",friction:"Ordinary foot friction floor",summary:"Training overview",step:"Iteration",raw:"Raw",smoothed:"Smoothed",resetTip:"Double-click to reset",failed:"Failed to read logs. Please refresh again.",warning:"Selected runs differ in initialization or friction settings. Their comparison does not isolate a single factor.",sidefoot:"Select up to 8 runs to compare. Colors match their curves. This page only reads logs.",help:"Hover for values; drag an interval to zoom all charts; double-click to reset. Faint lines are raw data; dark lines are smoothed.",footer:"Iteration indices start at zero; resumed training keeps checkpoint numbering. Curriculum is a mean across all environments. Slope-only levels were not logged. Training curves are not fixed-condition evaluations.",overview:"Overview",tracking:"Command tracking",feet:"Feet & gait",energy:"Energy & joints",recovery:"Fall recovery",diagnostics:"Training diagnostics",all:"All metrics"}
};
const state = {runs:[],selected:new Set(),group:"overview",query:"",data:null,zoom:null,charts:[],sequence:0,loading:false};
const t = key => {const zh=text.zh[key]||key,en=text.en[key]||key;return zh===en?zh:`${zh} / ${en}`;};
const name = run => run.title === run.title_en ? run.title : `${run.title} / ${run.title_en}`;
const metricName = metric => metric.name === metric.name_en ? metric.name : `${metric.name} / ${metric.name_en}`;
const metricHint = metric => [metric.hint,metric.hint_en].filter(Boolean).join("\n");
const unit = metric => metric.unit===metric.unit_en?metric.unit:`${metric.unit} / ${metric.unit_en}`;
const selectedRuns = () => state.runs.filter(run => state.selected.has(run.id));
const color = id => colors[[...state.selected].indexOf(id) % colors.length];
function element(tag, className, content) { const node=document.createElement(tag);if(className)node.className=className;if(content!==undefined)node.textContent=content;return node; }
function bilingual(node,zh,en){node.replaceChildren(document.createTextNode(zh));if(en&&en!==zh)node.append(element("span","bi-en",en));return node;}
function number(value) { if(!Number.isFinite(value))return "—";const a=Math.abs(value);return a>=1000?value.toLocaleString("zh-CN",{maximumFractionDigits:0}):a>=1?value.toFixed(3):a>=.01?value.toFixed(4):a>0?value.toPrecision(3):"0"; }
function tick(value) { const a=Math.abs(value);return a>=1000?Math.round(value).toLocaleString():a>=1?Number(value.toFixed(2)).toString():a>=.01?Number(value.toFixed(3)).toString():a>0?value.toExponential(1):"0"; }
function applyBilingual() {
 document.documentElement.lang="zh-CN";
 document.title="TQBot2 · 双语训练看板 / Bilingual Training Board";
 const ids={brandTitle:"brand",brandSubtitle:"subtitle",runsTitle:"runs",eyebrow:"eyebrow",refresh:"refresh",export:"export",smoothingLabel:"smoothing",axisLabel:"axis",autoLabel:"auto",resetZoom:"reset",chartHelp:"help",footer:"footer",loading:"loading",sidefoot:"sidefoot",empty:"empty"};
 for(const [id,key]of Object.entries(ids))$(id).textContent=t(key);
 for(const [id,key] of Object.entries({refresh:"refresh",export:"export",smoothingLabel:"smoothing",axisLabel:"axis",resetZoom:"reset"}))bilingual($(id),text.zh[key],text.en[key]);
 $("brandSubtitle").textContent="TQBot2 · TensorBoard";
 $("runSearch").placeholder="搜索训练 / Search runs";$("metricSearch").placeholder="搜索指标 / Search metrics: 课程 · torque…";
 $("runSearch").setAttribute("aria-label",t("runSearch"));$("metricSearch").setAttribute("aria-label",t("metricSearch"));$("tabs").setAttribute("aria-label",t("all"));
 for(const option of $("smoothing").options)option.textContent=option.value==="0"?t("off"):`${option.value} ${t("rounds")}`;
 for(const option of $("axis").options)option.textContent=t(option.value==="iteration"?"iteration":option.value==="relative"?"relative":"time");
 const recentKey=$("axis").value==="time"?"recentTime":"recent";bilingual($("recent"),text.zh[recentKey],text.en[recentKey]);
 renderTabs();renderRuns();renderSummary();renderCharts();updateTime();
}
function renderTabs(){ $("tabs").replaceChildren();for(const group of groups){const button=bilingual(element("button",state.group===group?"active":""),text.zh[group],text.en[group]);button.onclick=()=>{state.group=group;state.query="";$("metricSearch").value="";renderTabs();fetchCurves();};$("tabs").append(button);}bilingual($("pageTitle"),text.zh[state.group],text.en[state.group]); }
function renderRuns(){
 $("runList").replaceChildren();$("runCount").textContent=`${state.runs.length} ${t("records")}`;
 const query=$("runSearch").value.toLowerCase();
 for(const run of state.runs){if(query&&!`${run.title} ${run.title_en} ${run.name}`.toLowerCase().includes(query))continue;
  const row=element("label",`run-row${state.selected.has(run.id)?" selected":""}`);const check=element("input");check.type="checkbox";check.checked=state.selected.has(run.id);check.setAttribute("aria-label",name(run));
  check.onchange=()=>{if(check.checked){if(state.selected.size>=8){check.checked=false;$("error").textContent=t("max");return;}state.selected.add(run.id);}else state.selected.delete(run.id);localStorage.setItem("TQBot2-board-runs",JSON.stringify([...state.selected]));state.zoom=null;renderRuns();renderSummary();fetchCurves();};
  const details=element("div","run-details");const title=bilingual(element("div","run-title"),run.title,run.title_en);const dot=element("span","color-dot");dot.style.background=state.selected.has(run.id)?color(run.id):"#b9c1c8";title.prepend(dot);
  const date=element("div","run-date",run.date);const progress=element("div","run-progress");progress.append(element("span",`run-status ${run.status}`,t(run.status)),document.createTextNode(run.iteration===null?"":`${run.iteration.toLocaleString()} / ${run.target?run.target.toLocaleString():"—"}`));
  details.append(title,date,progress,element("div","run-name",run.name));row.append(check,details);$("runList").append(row);
 }
}
function renderSummary(){
 const runs=selectedRuns();$("selectionSummary").textContent=runs.length?`已选 ${runs.length} 组训练 / Comparing ${runs.length} runs · ${t("iteration")}`:t("select");$("runSummary").replaceChildren();
 for(const run of runs){const box=element("div","summary-box");const heading=bilingual(element("strong"),run.title,run.title_en);const dot=element("span","color-dot");dot.style.background=color(run.id);heading.prepend(dot);const big=element("div","big",run.iteration===null?"—":run.iteration.toLocaleString());big.append(element("small","",run.target?` / ${run.target.toLocaleString()}`:""));const footer=element("small","",`${t(run.status)}${run.friction_min===null||run.friction_min===undefined?"":` · μ ≥ ${run.friction_min}`}`);box.append(heading,big,footer);$("runSummary").append(box);}
 const differs=new Set(runs.map(run=>run.friction_min)).size>1||new Set(runs.map(run=>run.kind==="scratch"?"scratch":"inherited")).size>1;
 $("notice").style.display=runs.length>1&&differs?"block":"none";$("notice").textContent=t("warning");
}
function updateTime(){ $("liveState").textContent=t($("autoRefresh").checked?"live":"paused");if(state.data)$("updated").textContent=`${t("updated")} ${new Date(state.data.updated_at*1000).toLocaleTimeString("zh-CN",{hour12:false})}`; }
async function fetchRuns(initial=false){const response=await fetch("/api/runs");if(!response.ok)throw new Error("runs");const data=await response.json();state.runs=data.runs.sort((a,b)=>b.mtime-a.mtime);if(initial){let saved=[];try{saved=JSON.parse(localStorage.getItem("TQBot2-board-runs")||"[]");}catch{}state.selected=new Set(saved.filter(id=>state.runs.some(run=>run.id===id)).slice(0,8));if(!state.selected.size){const active=state.runs.find(run=>run.status==="running");const complete=state.runs.find(run=>run.kind==="resume"&&run.status==="completed")||state.runs.find(run=>run.status==="completed");for(const run of [active,complete])if(run)state.selected.add(run.id);if(!state.selected.size&&state.runs[0])state.selected.add(state.runs[0].id);}}renderRuns();renderSummary();}
async function fetchCurves(){
 const sequence=++state.sequence;$("error").textContent="";
 if(!state.selected.size){state.data=null;state.charts=[];$("charts").replaceChildren();$("loading").style.display="none";$("empty").hidden=false;$("empty").textContent=t("select");return;}
 if(!state.data)$("loading").style.display="block";$("refresh").disabled=true;
 try{const params=new URLSearchParams({runs:[...state.selected].join(","),group:state.group,q:state.query});const response=await fetch(`/api/scalars?${params}`);const data=await response.json();if(!response.ok)throw new Error(data.error);if(sequence!==state.sequence)return;state.data=data;renderCharts();updateTime();}
 catch(error){if(sequence===state.sequence)$("error").textContent=t("failed");console.error(error);}
 finally{if(sequence===state.sequence){$("loading").style.display="none";$("refresh").disabled=false;}}
}
function transform(points){if(!points.length)return [];const axis=$("axis").value;const first=points[0];return points.map(point=>[axis==="time"?(point[2]-first[2])/3600:axis==="relative"?point[0]-first[0]:point[0],point[1],point[0]]);}
function smooth(points,width){if(width<=1)return points;let sum=0;return points.map((point,i)=>{sum+=point[1];if(i>=width)sum-=points[i-width][1];return [point[0],sum/Math.min(width,i+1),point[2]];});}
function renderCharts(){
 state.charts=[];$("charts").replaceChildren();if(!state.data)return;$("empty").hidden=state.data.metrics.length>0;$("empty").textContent=t("empty");
 for(const metric of state.data.metrics){const card=element("section","chart-card");const heading=element("div","chart-title");const title=bilingual(element("h3"),metric.name,metric.name_en);title.title=metricHint(metric);heading.append(title);if(metric.direction!=="neutral")heading.append(bilingual(element("span","direction"),text.zh[metric.direction],text.en[metric.direction]));const tag=element("div","tag",metric.tag);tag.title=`${metric.tag}\n${metricHint(metric)}`;const canvas=element("canvas","chart-canvas");canvas.setAttribute("aria-label",metricName(metric));const tooltip=element("div","tooltip");const legend=element("div","legend");card.append(heading,tag,canvas,legend,tooltip);$("charts").append(card);
  const traces=state.data.runs.filter(data=>state.selected.has(data.id)).map(data=>({run:state.runs.find(run=>run.id===data.id),points:transform(data.values[metric.tag]||[])})).filter(trace=>trace.run);
  for(const trace of traces){const item=element("span","legend-item");const dot=element("span","color-dot");dot.style.background=color(trace.run.id);item.append(dot,bilingual(element("span","legend-name"),trace.run.title,trace.run.title_en),element("strong","",trace.points.length?number(trace.points.at(-1)[1]):t("notlogged")));legend.append(item);}
  const chart={card,canvas,tooltip,metric,traces,hover:null,drag:null,geometry:null};state.charts.push(chart);bindChart(chart);draw(chart);
 }
}
function draw(chart){
 const {canvas,traces,metric}=chart;const w=canvas.clientWidth,h=canvas.clientHeight,dpr=window.devicePixelRatio||1;canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);if(w<80)return;
 const box={left:52,right:w-12,top:16,bottom:h-29};let low=Infinity,high=-Infinity;for(const trace of traces)for(const point of trace.points){low=Math.min(low,point[0]);high=Math.max(high,point[0]);}if(!Number.isFinite(low))return;let xmin=state.zoom?state.zoom[0]:low,xmax=state.zoom?state.zoom[1]:high;if(xmax<=xmin)xmax=xmin+1;
 let ymin=Infinity,ymax=-Infinity;for(const trace of traces)for(const point of trace.points)if(point[0]>=xmin&&point[0]<=xmax){ymin=Math.min(ymin,point[1]);ymax=Math.max(ymax,point[1]);}
 if(!Number.isFinite(ymin)){ymin=0;ymax=1;}const padding=(ymax-ymin||Math.abs(ymax)||1)*.08;ymin-=padding;ymax+=padding;
 const px=x=>box.left+(x-xmin)/(xmax-xmin)*(box.right-box.left),py=y=>box.bottom-(y-ymin)/(ymax-ymin)*(box.bottom-box.top);chart.geometry={box,xmin,xmax,px,py,w,h};
 ctx.font='10px "Noto Sans CJK SC",sans-serif';ctx.fillStyle="#89959e";ctx.strokeStyle="#eaf0f4";ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;ctx.beginPath();ctx.moveTo(box.left,py(y));ctx.lineTo(box.right,py(y));ctx.stroke();ctx.textAlign="right";ctx.fillText(tick(y),box.left-7,py(y)+3);}
 for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4;ctx.textAlign="center";ctx.fillText(tick(x),px(x),h-10);}
 if(unit(metric)){ctx.textAlign="left";ctx.fillStyle="#9ba5ac";ctx.fillText(unit(metric),box.left,10);}
 ctx.save();ctx.beginPath();ctx.rect(box.left,box.top,box.right-box.left,box.bottom-box.top);ctx.clip();
 const width=Number($("smoothing").value);
 for(const trace of traces){if(!trace.points.length)continue;const smoothed=smooth(trace.points,width);trace.smoothed=smoothed;
  for(const [points,alpha,lineWidth] of width>1?[[trace.points,.17,.75],[smoothed,1,1.7]]:[[trace.points,1,1.35]]){ctx.strokeStyle=color(trace.run.id);ctx.globalAlpha=alpha;ctx.lineWidth=lineWidth;ctx.beginPath();let started=false;for(const point of points){if(point[0]<xmin||point[0]>xmax)continue;if(!started){ctx.moveTo(px(point[0]),py(point[1]));started=true;}else ctx.lineTo(px(point[0]),py(point[1]));}ctx.stroke();}
 }
 ctx.globalAlpha=1;
 if(chart.hover!==null){ctx.strokeStyle="#8799a5";ctx.setLineDash([3,3]);ctx.beginPath();ctx.moveTo(chart.hover,box.top);ctx.lineTo(chart.hover,box.bottom);ctx.stroke();ctx.setLineDash([]);}
 if(chart.drag){ctx.fillStyle="#087e8618";ctx.fillRect(Math.min(...chart.drag),box.top,Math.abs(chart.drag[1]-chart.drag[0]),box.bottom-box.top);}
 ctx.restore();
}
function nearest(points,x){let lo=0,hi=points.length-1;while(lo<hi){const mid=Math.floor((lo+hi)/2);if(points[mid][0]<x)lo=mid+1;else hi=mid;}if(lo>0&&Math.abs(points[lo-1][0]-x)<Math.abs(points[lo][0]-x))return lo-1;return lo;}
function bindChart(chart){const canvas=chart.canvas;
 canvas.onpointermove=event=>{if(!chart.geometry)return;const rect=canvas.getBoundingClientRect();const point=Math.max(chart.geometry.box.left,Math.min(chart.geometry.box.right,event.clientX-rect.left));chart.hover=point;if(chart.drag){chart.drag[1]=point;draw(chart);return;}draw(chart);const x=chart.geometry.xmin+(point-chart.geometry.box.left)/(chart.geometry.box.right-chart.geometry.box.left)*(chart.geometry.xmax-chart.geometry.xmin);chart.tooltip.replaceChildren();chart.tooltip.append(element("b","",`${t($("axis").value==="time"?"hours":"step")} ${tick(x)}`));for(const trace of chart.traces){if(!trace.points.length||x<trace.points[0][0]||x>trace.points.at(-1)[0])continue;const i=nearest(trace.points,x);const row=element("div");const dot=element("span","color-dot");dot.style.background=color(trace.run.id);row.append(dot,document.createTextNode(`${name(trace.run)}: ${number(trace.points[i][1])}`));chart.tooltip.append(row);if(Number($("smoothing").value)>1)chart.tooltip.append(element("div","",`${t("smoothed")}: ${number(trace.smoothed[i][1])}`));}chart.tooltip.style.display="block";chart.tooltip.style.left=`${Math.max(8,Math.min(point+20,chart.card.clientWidth-chart.tooltip.offsetWidth-12))}px`;chart.tooltip.style.top=`${Math.max(65,Math.min(event.clientY-chart.card.getBoundingClientRect().top-10,chart.card.clientHeight-chart.tooltip.offsetHeight-12))}px`;};
 canvas.onpointerleave=()=>{if(chart.drag)return;chart.hover=null;chart.tooltip.style.display="none";draw(chart);};
 canvas.onpointerdown=event=>{if(!chart.geometry)return;const point=event.clientX-canvas.getBoundingClientRect().left;if(point<chart.geometry.box.left||point>chart.geometry.box.right)return;canvas.setPointerCapture(event.pointerId);chart.drag=[point,point];chart.tooltip.style.display="none";};
 canvas.onpointerup=()=>{if(!chart.drag)return;const [a,b]=chart.drag;chart.drag=null;if(Math.abs(a-b)>12){const {box,xmin,xmax}=chart.geometry;const convert=x=>xmin+(x-box.left)/(box.right-box.left)*(xmax-xmin);state.zoom=[convert(Math.min(a,b)),convert(Math.max(a,b))];}chart.hover=null;for(const item of state.charts)draw(item);};
 canvas.ondblclick=()=>{state.zoom=null;for(const item of state.charts)draw(item);};
}
function exportCSV(){const params=new URLSearchParams({runs:[...state.selected].join(","),group:state.group,q:state.query,axis:$("axis").value});if(state.zoom){params.set("low",state.zoom[0]);params.set("high",state.zoom[1]);}$("export").href=`/api/export.csv?${params}`;}

let debounce;
$("metricSearch").oninput=()=>{clearTimeout(debounce);debounce=setTimeout(()=>{state.query=$("metricSearch").value.trim();fetchCurves();},300);};
$("runSearch").oninput=renderRuns;
$("smoothing").onchange=()=>{for(const chart of state.charts)draw(chart);};
$("axis").onchange=()=>{state.zoom=null;applyBilingual();};
$("autoRefresh").onchange=updateTime;
$("resetZoom").onclick=()=>{state.zoom=null;for(const chart of state.charts)draw(chart);};
$("recent").onclick=()=>{let max=-Infinity;for(const chart of state.charts)for(const trace of chart.traces)if(trace.points.length)max=Math.max(max,trace.points.at(-1)[0]);if(!Number.isFinite(max))return;state.zoom=[Math.max(0,max-($("axis").value==="time"?.5:500)),max];for(const chart of state.charts)draw(chart);};
$("export").onclick=exportCSV;
$("refresh").onclick=async()=>{try{await fetchRuns();await fetchCurves();}catch(error){$("error").textContent=t("failed");console.error(error);}};
window.addEventListener("resize",()=>{for(const chart of state.charts)draw(chart);});
setInterval(async()=>{if(!$("autoRefresh").checked)return;try{await fetchRuns();await fetchCurves();}catch(error){$("error").textContent=t("failed");console.error(error);}},20000);
(async()=>{applyBilingual();try{await fetchRuns(true);await fetchCurves();}catch(error){$("loading").style.display="none";$("error").textContent=t("failed");console.error(error);}})();
