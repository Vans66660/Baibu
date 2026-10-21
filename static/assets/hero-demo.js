(() => {
  'use strict';
  const surface = document.getElementById('heroMotion');
  const frame = document.getElementById('heroAnimation');
  const control = document.getElementById('heroPlay');
  const caption = document.getElementById('heroDemoCaption');
  if (!surface || !frame || !control) return;
  const ctx = surface.getContext('2d');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const phaseLabels = ['01 点选椅子', '02 拖动位置 · 原处背景已补全', '03 独立调整宽高'];
  const steps = [...document.querySelectorAll('.demo-steps i')];
  const ease = v => { v = Math.max(0, Math.min(1, v)); return v * v * (3 - 2 * v); };
  const mix = (a, b, progress) => a + (b - a) * ease(progress);
  const image = url => new Promise((resolve, reject) => {
    const img = new Image(); img.onload = () => resolve(img); img.onerror = reject; img.src = url;
  });
  let running = !reducedMotion.matches;
  let visible = true;
  let raf = null;
  let elapsed = 0;
  let last = null;
  let render = null;
  let phase = -1;

  function syncControl() {
    control.setAttribute('aria-label', running ? '暂停功能演示' : '播放功能演示');
    control.setAttribute('aria-pressed', String(!running));
    control.firstElementChild.textContent = running ? 'Ⅱ' : '▶';
  }
  function tick(now) {
    raf = null;
    if (!running || !visible || document.hidden) { last = null; return; }
    if (last != null) elapsed += Math.min(now - last, 80);
    last = now;
    render((elapsed / 1000) % 14);
    raf = requestAnimationFrame(tick);
  }
  function schedule() {
    if (render && running && visible && !document.hidden && raf == null) {
      last = null; raf = requestAnimationFrame(tick);
    }
  }
  function pause() {
    if (raf != null) cancelAnimationFrame(raf);
    raf = null; last = null;
  }
  control.addEventListener('click', () => {
    running = !running;
    syncControl();
    if (running) schedule(); else pause();
  });
  reducedMotion.addEventListener('change', event => {
    running = !event.matches;
    syncControl();
    if (running) schedule(); else pause();
  });
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    if (visible) schedule(); else pause();
  }, {threshold: 0.05}).observe(frame);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) pause(); else schedule();
  });

  Promise.all([
    image('/assets/hero-background.webp'),
    image('/assets/hero-chair.webp'),
    fetch('/assets/hero-scene.json').then(response => {
      if (!response.ok) throw new Error('Preview unavailable');
      return response.json();
    }),
  ]).then(([background, object, scene]) => {
    surface.width = scene.width;
    surface.height = scene.height;
    const [x0, y0, w0, h0] = scene.crop;
    const bottom = y0 + h0;
    const move = Math.min(80, x0 * 0.32);
    function drawCursor(x, y, resize = false) {
      ctx.save(); ctx.translate(x, y);
      ctx.shadowColor = '#15091b'; ctx.shadowBlur = 7;
      ctx.lineJoin = 'round'; ctx.lineWidth = 2;
      ctx.strokeStyle = '#412349'; ctx.fillStyle = '#fff5e5';
      ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(2,26);ctx.lineTo(9,19);ctx.lineTo(16,31);ctx.lineTo(21,28);ctx.lineTo(14,16);ctx.lineTo(25,14);ctx.closePath();
      ctx.fill();ctx.stroke();
      if (resize) { ctx.shadowBlur=0;ctx.font='17px sans-serif';ctx.fillStyle='#fff4d6';ctx.fillText('↔',24,30); }
      ctx.restore();
    }
    render = t => {
      let x=x0, sx=1, sy=1;
      if (t >= 2.3 && t < 4.8) x=mix(x0,x0-move,(t-2.3)/2.5);
      else if (t >= 4.8 && t < 11.8) x=x0-move;
      if (t >= 6.8 && t < 8.6) sx=mix(1,1.18,(t-6.8)/1.8);
      else if (t >= 8.6 && t < 11.8) sx=1.18;
      if (t >= 8.6 && t < 10.3) sy=mix(1,.8,(t-8.6)/1.7);
      else if (t >= 10.3 && t < 11.8) sy=.8;
      if(t>=11.8){const r=(t-11.8)/1.9;x=mix(x0-move,x0,r);sx=mix(1.18,1,r);sy=mix(.8,1,r);}
      const w=w0*sx, h=h0*sy, y=bottom-h;
      const currentPhase=t<2.3?0:t<6.8?1:2;
      if(currentPhase!==phase){phase=currentPhase;caption.textContent=phaseLabels[phase];steps.forEach((step,i)=>step.classList.toggle('active',i===phase));}
      ctx.clearRect(0,0,surface.width,surface.height);
      ctx.drawImage(background,0,0);
      ctx.save();ctx.fillStyle='#211817';ctx.globalAlpha=.19;ctx.filter='blur(5px)';
      ctx.beginPath();ctx.ellipse(x+w*.5,bottom-2,w*.44,8*sy,0,0,Math.PI*2);ctx.fill();ctx.restore();
      ctx.drawImage(object,x,y,w,h);
      const selected=t>=1.05&&t<13.7;
      if(selected){
        const gap=5;
        ctx.save();ctx.strokeStyle='#ffe0a7';ctx.lineWidth=1.6;ctx.shadowColor='#ebc68e';ctx.shadowBlur=6;
        ctx.strokeRect(x-gap,y-gap,w+gap*2,h+gap*2);
        ctx.shadowBlur=0;
        [[x-gap,y-gap],[x+w+gap,y-gap],[x-gap,y+h+gap],[x+w+gap,y+h+gap]].forEach(([hx,hy])=>{
          ctx.fillStyle='#ffe7bd';ctx.fillRect(hx-4,hy-4,8,8);ctx.strokeStyle='#755078';ctx.lineWidth=1;ctx.strokeRect(hx-4,hy-4,8,8);
        });
        const text=t<6.8?'✦ 椅子已选中':`宽 ${Math.round(sx*100)}% · 高 ${Math.round(sy*100)}%`;
        ctx.font='15px -apple-system, sans-serif';
        const labelWidth=ctx.measureText(text).width+20;
        ctx.fillStyle='#29152f';ctx.fillRect(x-gap,y-gap-34,labelWidth,27);
        ctx.fillStyle='#ffe0ae';ctx.fillText(text,x+5,y-gap-15);
        ctx.restore();
      }
      let px=x+w*.59,py=y+h*.52;
      if(t<1.05){px=mix(x0+w0+70,x0+w0*.59,t/1.05);py=mix(y0-40,y0+h0*.52,t/1.05);}
      if(t>=4.8&&t<6.8){px=mix(x+w*.59,x+w+5,(t-4.8)/1.5);py=mix(y+h*.52,y-5,(t-4.8)/1.5);}
      if(t>=6.8&&t<11.8){px=x+w+5;py=y-5;}
      if(t>=11.8){px=x+w*.59;py=y+h*.52;}
      if(t>=.85&&t<1.45){const p=(t-.85)/.6;ctx.save();ctx.globalAlpha=1-p;ctx.strokeStyle='#ffe2b4';ctx.lineWidth=2;ctx.beginPath();ctx.arc(px,py,8+p*22,0,Math.PI*2);ctx.stroke();ctx.restore();}
      drawCursor(px,py,t>=6.8&&t<11.8);
    };
    frame.classList.add('motion-ready');
    frame.setAttribute('aria-label','自动循环功能演示，可使用右上角按钮暂停');
    control.hidden=false;
    syncControl();
    render(reducedMotion.matches?2:0);
    schedule();
  }).catch(() => {
    // Keep the original static scene available if a demo asset cannot load.
    control.hidden=true;
  });
})();
